use crate::println;
use crate::comm::uart;
use crate::utils::memory;
    
pub fn entrypoint() {
    uart::init();
    // initialize memory
    memory::init_heap();
    // call the user's main
    crate::main();
    // this has to be here otherwise the stage 1 bootloader won't think we are done
    println!("DONE!!!");
}
