((nil . ((eval . (progn
                   (use-package lispi
                     :straight nil
                     :load-path "/Users/houjun/Documents/Projects/lispi/"
                     :custom
                     (lispi-binary "/Users/houjun/Documents/Projects/lispi/target/debug/unix-side")
                     :config
                     (evil-leader/set-key-for-mode 'lisp-mode
                       "ht" 'lispi-eval-last-sexp
                       "ue" 'lispi-eval-last-sexp
                       "hn" 'lispi-eval-defun
                       "hb" 'lispi-eval-buffer
                       "hc" 'lispi-remove-overlays
                       "hst" 'lispi-connect
                       "hsp" 'lispi-disconnect))
                   (add-to-list 'auto-mode-alist '("\\.lispi\\'" . lisp-mode))
                   (add-hook 'lisp-mode-hook
                             (lambda ()
                               (when (and buffer-file-name
                                          (string-match-p "\\.lispi\\'" buffer-file-name))
                                 (sly-mode -1)
                                 (lispi-mode 1)))
                             nil t))))))
