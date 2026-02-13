__WORD_ALIGN = 4;
__STACK_ALIGN = 8;
__CACHE_ALIGN = 32;

__BOOTLOADER_LOAD_ADDR = 0x00008000;
__STACK_INIT = 0x08000000;
__DRAM_END   = 0x20000000;

ENTRY(_start)
PROVIDE(__stack_init__ = __STACK_INIT);
PROVIDE(__dram_end__ = __DRAM_END);

SECTIONS {
    .text __BOOTLOADER_LOAD_ADDR : {
        PROVIDE(__code_start__ = .);
        . = ALIGN(__WORD_ALIGN);
        KEEP(*(.text.boot))
        *(.text*)
        PROVIDE(__code_end__ = .);
        . = ALIGN(__WORD_ALIGN);
    }
    .rodata ALIGN(__WORD_ALIGN) : {
        *(.rodata*)
        . = ALIGN(__WORD_ALIGN);
    }
    .data ALIGN(__WORD_ALIGN) : {
        *(.data*)
        . = ALIGN(__WORD_ALIGN);
    }
    .bss ALIGN(__WORD_ALIGN) : {
        PROVIDE(__bss_start__ = .);
        *(.bss*)
        *(COMMON)
        PROVIDE(__bss_end__ = .);

        . = ALIGN(__WORD_ALIGN);
        PROVIDE(__prog_end__ = .);

        /* high alignment so that the heap allocator can make more assumptions */
        . = ALIGN(__CACHE_ALIGN);
        PROVIDE(__heap_start__ = .);
    }
}
