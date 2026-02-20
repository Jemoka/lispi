use crate::comm::uart;
use crate::println;
use crate::utils::exceptions;
use crate::utils::memory;

pub fn entrypoint() {
    // initialize uart
    uart::init();
    // initialze exception vectors
    exceptions::activate_exception_vector_hook();
    // initialize memory
    memory::init_heap();
    // call the user's main
    crate::main();
    // this has to be here otherwise the stage 1 bootloader won't think we are done
    println!("DONE!!!");
}
