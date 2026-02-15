use crate::println;
use crate::comm::uart;
    
pub fn entrypoint() {
    uart::init();
    // call the user's main
    crate::main();
    // this has to be here otherwise the stage 1 bootloader won't think we are done
    println!("DONE!!!");
}
