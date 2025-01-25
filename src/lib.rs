pub mod classifier;

pub use classifier::Classifier;

pub fn init_logger() {
    env_logger::init();
}
