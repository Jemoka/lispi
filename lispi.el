;;; lispi.el --- Evaluate Lisp on a Raspberry Pi -*- lexical-binding: t; -*-

;; Author: houjun
;; Version: 0.1
;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; Emacs interface to the lispi unix-side REPL binary.  Spawns the
;; binary as a subprocess, sends Lisp expressions over stdin, and
;; displays results as inline overlays.
;;
;; Setup with use-package:
;;
;;   (use-package lispi
;;     :load-path "/path/to/lispi"
;;     :custom
;;     (lispi-binary "/path/to/lispi/target/debug/unix-side")
;;     :hook (lisp-mode . lispi-mode)
;;     :commands (lispi-connect lispi-disconnect))
;;
;; Variables:
;;   `lispi-binary'        — path to the compiled unix-side binary (required)
;;   `lispi-overlay-prefix' — string prepended to results (default "=> ")

;;; Code:

;;;; Customization

(defgroup lispi nil
  "Evaluate Lisp expressions on a Raspberry Pi."
  :group 'tools
  :prefix "lispi-")

(defcustom lispi-binary "unix-side"
  "Path to the unix-side REPL binary."
  :type 'string
  :group 'lispi)

(defcustom lispi-overlay-prefix "=> "
  "Prefix shown before result overlays."
  :type 'string
  :group 'lispi)

;;;; Faces

(defface lispi-eval-overlay
  '((t :foreground "RoyalBlue" :weight bold))
  "Face for inline evaluation result overlays."
  :group 'lispi)

(defface lispi-eval-overlay-folded
  '((t :foreground "gray50" :weight normal))
  "Face for the fold indicator on multi-line results."
  :group 'lispi)

;;;; Process management

(defvar lispi--process nil
  "The subprocess running the unix-side binary.")

(defvar lispi--output-buffer ""
  "Accumulator for partial output from the subprocess.")

(defvar lispi--callback nil
  "Pending callback to invoke with the next complete result.")

(defvar lispi--ready nil
  "Non-nil once the initial prompt has been received.")

(defun lispi--process-filter (_proc output)
  "Accumulate OUTPUT from the subprocess and dispatch when complete."
  (setq lispi--output-buffer (concat lispi--output-buffer output))
  ;; A complete response ends with "\n> " (result + prompt).
  ;; Also match bare "> " for the initial startup prompt.
  (when (string-match-p "\n> \\'" lispi--output-buffer)
    (let ((text (string-trim
                 (replace-regexp-in-string "\n> \\'" "" lispi--output-buffer))))
      (setq lispi--output-buffer "")
      (if lispi--ready
          (when lispi--callback
            (let ((cb lispi--callback))
              (setq lispi--callback nil)
              (funcall cb text)))
        (setq lispi--ready t)
        (message "lispi: connected"))))
  ;; Bare initial prompt (no preceding newline).
  (when (and (not lispi--ready)
             (string-match-p "\\`> \\'" lispi--output-buffer))
    (setq lispi--output-buffer "")
    (setq lispi--ready t)
    (message "lispi: connected")))

(defun lispi--sentinel (_proc event)
  "Handle process EVENT."
  (setq lispi--ready nil)
  (setq lispi--callback nil)
  (setq lispi--output-buffer "")
  (message "lispi: %s" (string-trim event)))

(defun lispi-connect ()
  "Start the unix-side REPL subprocess."
  (interactive)
  (when (and lispi--process (process-live-p lispi--process))
    (user-error "lispi: already connected"))
  (setq lispi--ready nil
        lispi--output-buffer ""
        lispi--callback nil)
  (let ((proc (start-process "lispi" nil lispi-binary)))
    (set-process-filter proc #'lispi--process-filter)
    (set-process-sentinel proc #'lispi--sentinel)
    (set-process-query-on-exit-flag proc nil)
    (setq lispi--process proc))
  (message "lispi: starting %s …" lispi-binary))

(defun lispi-disconnect ()
  "Kill the unix-side subprocess."
  (interactive)
  (when (and lispi--process (process-live-p lispi--process))
    (kill-process lispi--process))
  (setq lispi--process nil
        lispi--ready nil))

;;;; Sending expressions

(defun lispi--strip-comments (str)
  "Remove ; line-comments from STR, respecting string literals."
  (let ((lines (split-string str "\n")))
    (mapconcat
     (lambda (line)
       (let ((i 0) (in-str nil) (len (length line)))
         (while (< i len)
           (let ((ch (aref line i)))
             (cond
              ((and (not in-str) (= ch ?\;))
               (setq len i))
              ((= ch ?\")
               (setq in-str (not in-str))
               (setq i (1+ i)))
              ((and in-str (= ch ?\\))
               (setq i (+ i 2)))
              (t (setq i (1+ i))))))
         (substring line 0 len)))
     lines " ")))

(defun lispi--send (expr callback)
  "Send EXPR string to the subprocess, call CALLBACK with the result."
  (unless (and lispi--process (process-live-p lispi--process))
    (user-error "lispi: not connected — run M-x lispi-connect"))
  (unless lispi--ready
    (user-error "lispi: waiting for initial prompt"))
  (when lispi--callback
    (user-error "lispi: evaluation already in progress"))
  (setq lispi--callback callback)
  ;; The binary reads one line at a time, so collapse multi-line
  ;; expressions into a single line.  Strip ; comments first so they
  ;; don't swallow the rest of the expression after newline removal.
  (let ((one-line (string-trim (lispi--strip-comments expr))))
    (process-send-string lispi--process (concat one-line "\n"))))

;;;; Overlays

(defvar-local lispi--overlays nil
  "List of active result overlays in this buffer.")

(defun lispi--display-result (pos result)
  "Show RESULT as an overlay after POS in the current buffer.
Removes any existing lispi overlay at POS first."
  ;; Remove any prior lispi overlay at or near this position.
  ;; We scan our own list since overlays-at misses zero-width overlays.
  (dolist (ov (copy-sequence lispi--overlays))
    (when (and (overlay-buffer ov)
               (<= (abs (- (overlay-start ov) pos)) 1))
      (lispi--remove-overlay ov)))
  (let* ((multiline (string-match-p "\n" result))
         (display-text (if multiline
                          (concat lispi-overlay-prefix
                                  (car (split-string result "\n"))
                                  " [+]")
                        (concat lispi-overlay-prefix result)))
         (ov (make-overlay pos pos nil nil nil)))
    (overlay-put ov 'lispi-result t)
    (overlay-put ov 'lispi-full-result result)
    (overlay-put ov 'lispi-folded multiline)
    (overlay-put ov 'after-string
                 (propertize (concat "  " display-text)
                             'face (if multiline
                                       'lispi-eval-overlay-folded
                                     'lispi-eval-overlay)))
    ;; Evaporate on modification.
    (overlay-put ov 'modification-hooks
                 (list (lambda (o &rest _) (lispi--remove-overlay o))))
    (overlay-put ov 'insert-in-front-hooks
                 (list (lambda (o &rest _) (lispi--remove-overlay o))))
    (push ov lispi--overlays)
    ov))

(defun lispi--remove-overlay (ov)
  "Delete overlay OV and remove it from the local list."
  (setq lispi--overlays (delq ov lispi--overlays))
  (delete-overlay ov))

(defun lispi-toggle-fold-at-point ()
  "Toggle fold/unfold of a multi-line result overlay at point."
  (interactive)
  (let ((ov (seq-find (lambda (o) (overlay-get o 'lispi-result))
                      (overlays-at (point)))))
    (unless ov (user-error "No lispi result overlay at point"))
    (let* ((result (overlay-get ov 'lispi-full-result))
           (folded (overlay-get ov 'lispi-folded))
           (new-folded (not folded))
           (display-text (if new-folded
                             (concat lispi-overlay-prefix
                                     (car (split-string result "\n"))
                                     " [+]")
                           (concat lispi-overlay-prefix result))))
      (overlay-put ov 'lispi-folded new-folded)
      (overlay-put ov 'after-string
                   (propertize (concat "  " display-text)
                               'face (if new-folded
                                         'lispi-eval-overlay-folded
                                       'lispi-eval-overlay))))))

(defun lispi-remove-overlays ()
  "Remove all lispi result overlays from the current buffer."
  (interactive)
  (dolist (ov lispi--overlays)
    (delete-overlay ov))
  (setq lispi--overlays nil))

;;;; Eval commands

(defun lispi--sexp-bounds-before-point ()
  "Return (BEG . END) of the enclosing top-level form or the sexp at point.
When inside a paren form, grabs the whole top-level form.
When at top level (bare atom), grabs the sexp at/before point."
  (save-excursion
    (if (> (nth 0 (syntax-ppss)) 0)
        ;; Inside a paren form — grab the top-level form
        (progn
          (beginning-of-defun)
          (let ((beg (point)))
            (forward-sexp)
            (cons beg (point))))
      ;; At top level — grab sexp at/before point
      (backward-sexp)
      (let ((beg (point)))
        (forward-sexp)
        (cons beg (point))))))

(defun lispi--sexp-bounds-at-point ()
  "Return (BEG . END) of the sexp at point.
Uses `forward-sexp'/`backward-sexp' for all cases.  When point is
on or just after a closing paren, scans backward first."
  (save-excursion
    (cond
     ((or (and (char-before) (eq (char-syntax (char-before)) ?\)))
          (and (char-after) (eq (char-syntax (char-after)) ?\))))
      (when (and (char-after) (eq (char-syntax (char-after)) ?\)))
        (forward-char))
      (let ((end (point)))
        (backward-sexp)
        (cons (point) end)))
     (t
      (forward-sexp)
      (let ((end (point)))
        (backward-sexp)
        (cons (point) end))))))

(defun lispi--eval-and-overlay (beg end)
  "Send buffer text between BEG and END, display result overlay at END."
  (let ((expr (buffer-substring-no-properties beg end))
        (pos end)
        (buf (current-buffer)))
    (lispi--send expr
                 (lambda (result)
                   (with-current-buffer buf
                     (lispi--display-result pos result))))))

(defun lispi-eval-last-sexp ()
  "Evaluate the sexp before point and show the result as an overlay."
  (interactive)
  (let ((bounds (lispi--sexp-bounds-before-point)))
    (lispi--eval-and-overlay (car bounds) (cdr bounds))))

(defun lispi-eval-at-point ()
  "Evaluate the sexp at point and show the result as an overlay."
  (interactive)
  (let ((bounds (lispi--sexp-bounds-at-point)))
    (unless bounds (user-error "No sexp at point"))
    (lispi--eval-and-overlay (car bounds) (cdr bounds))))

(defun lispi-eval-defun ()
  "Evaluate the top-level form around point."
  (interactive)
  (save-excursion
    (end-of-defun)
    (let ((end (point)))
      (beginning-of-defun)
      (lispi--eval-and-overlay (point) end))))

(defun lispi-eval-region (beg end)
  "Evaluate the region between BEG and END."
  (interactive "r")
  (lispi--eval-and-overlay beg end))

(defun lispi-eval-line-or-region ()
  "Evaluate the active region, or the current line if no region."
  (interactive)
  (if (use-region-p)
      (lispi--eval-and-overlay (region-beginning) (region-end))
    (lispi--eval-and-overlay (line-beginning-position) (line-end-position))))

(defun lispi--collect-top-level-sexps ()
  "Return a list of (BEG . END) for each top-level sexp in the buffer."
  (save-excursion
    (goto-char (point-min))
    (let (sexps)
      (while (let ((pos (point)))
               (forward-comment (point-max))
               (not (eobp)))
        (let ((beg (point))
              (end (scan-sexps (point) 1)))
          (unless end (user-error "Unbalanced sexp at position %d" beg))
          (push (cons beg end) sexps)
          (goto-char end)))
      (nreverse sexps))))

(defun lispi-eval-buffer ()
  "Evaluate every top-level sexp in the buffer sequentially."
  (interactive)
  (let* ((buf (current-buffer))
         (sexps (lispi--collect-top-level-sexps))
         (total (length sexps)))
    (when (zerop total)
      (user-error "No sexps found in buffer"))
    (message "lispi: evaluating %d forms…" total)
    (let ((idx 0))
      (cl-labels
          ((eval-next ()
             (if (>= idx total)
                 (message "lispi: done (%d forms)" total)
               (let* ((bounds (nth idx sexps))
                      (beg (car bounds))
                      (end (cdr bounds))
                      (expr (with-current-buffer buf
                              (buffer-substring-no-properties beg end))))
                 (cl-incf idx)
                 (lispi--send expr
                              (lambda (result)
                                (with-current-buffer buf
                                  (lispi--display-result end result))
                                (eval-next)))))))
        (eval-next)))))

(defun lispi-eval-string (expr)
  "Evaluate EXPR entered in the minibuffer."
  (interactive "sEval: ")
  (lispi--send expr
               (lambda (result)
                 (message "lispi: %s%s" lispi-overlay-prefix result))))

;;;; Minor mode

(defvar lispi-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "C-c C-e") #'lispi-eval-last-sexp)
    (define-key map (kbd "C-c C-b") #'lispi-eval-defun)
    (define-key map (kbd "C-c C-p") #'lispi-eval-at-point)
    (define-key map (kbd "C-c C-r") #'lispi-eval-region)
    (define-key map (kbd "C-c C-l") #'lispi-eval-line-or-region)
    (define-key map (kbd "C-c C-c") #'lispi-eval-string)
    (define-key map (kbd "C-c C-a") #'lispi-eval-buffer)
    (define-key map (kbd "C-c C-k") #'lispi-remove-overlays)
    (define-key map (kbd "C-c C-d") #'lispi-disconnect)
    (define-key map (kbd "C-c C-f") #'lispi-toggle-fold-at-point)
    map)
  "Keymap for `lispi-mode'.")

;;;###autoload
(define-minor-mode lispi-mode
  "Minor mode for evaluating Lisp via the lispi unix-side REPL."
  :lighter " lispi"
  :keymap lispi-mode-map
  (if lispi-mode
      (add-hook 'after-change-functions #'lispi--after-change nil t)
    (remove-hook 'after-change-functions #'lispi--after-change t)
    (lispi-remove-overlays)))

(defun lispi--after-change (beg end _len)
  "Remove overlays that overlap the changed region BEG..END."
  (dolist (ov (overlays-in beg end))
    (when (overlay-get ov 'lispi-result)
      (lispi--remove-overlay ov))))

(provide 'lispi)
;;; lispi.el ends here
