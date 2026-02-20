//! critical section implementation for our pi

use crate::regs::cpsr::PSR;
use critical_section::RawRestoreState;

struct Cs;
critical_section::set_impl!(Cs);

unsafe impl critical_section::Impl for Cs {
    unsafe fn acquire() -> RawRestoreState {
        let cpsr = PSR::get_cpsr();

        cpsr.with_irq_disable(true)
            .with_fiq_disable(true)
            .set_cpsr();

        cpsr.into()
    }

    unsafe fn release(token: RawRestoreState) {
        PSR::from(token).set_cpsr();
    }
}
