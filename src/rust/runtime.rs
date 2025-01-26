use lazy_static::lazy_static;
use ort::{Environment, GraphOptimizationLevel, OrtError};
use std::sync::Arc;

lazy_static! {
    static ref ONNX_ENV: Arc<Environment> = init_onnx_environment()
        .expect("Failed to initialize ONNX Runtime environment");
}

#[derive(Debug)]
pub struct RuntimeConfig {
    pub inter_threads: i16,
    pub intra_threads: i16,
    pub optimization_level: GraphOptimizationLevel,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            inter_threads: 0, // Let ONNX Runtime decide
            intra_threads: 0, // Let ONNX Runtime decide
            optimization_level: GraphOptimizationLevel::Level3,
        }
    }
}

impl Clone for RuntimeConfig {
    fn clone(&self) -> Self {
        Self {
            inter_threads: self.inter_threads,
            intra_threads: self.intra_threads,
            optimization_level: match self.optimization_level {
                GraphOptimizationLevel::Level1 => GraphOptimizationLevel::Level1,
                GraphOptimizationLevel::Level2 => GraphOptimizationLevel::Level2,
                GraphOptimizationLevel::Level3 => GraphOptimizationLevel::Level3,
                GraphOptimizationLevel::Disable => GraphOptimizationLevel::Disable,
            },
        }
    }
}

fn init_onnx_environment() -> Result<Arc<Environment>, OrtError> {
    let builder = Environment::builder()
        .with_name("prefrontal")
        .with_log_level(ort::LoggingLevel::Warning);

    Ok(Arc::new(builder.build()?))
}

pub fn get_env() -> Arc<Environment> {
    ONNX_ENV.clone()
}

pub fn create_session_builder(config: &RuntimeConfig) -> Result<ort::SessionBuilder, OrtError> {
    let mut builder = ort::SessionBuilder::new(&get_env())?;

    // Configure threading
    if config.inter_threads > 0 {
        builder = builder.with_inter_threads(config.inter_threads)?;
    }
    if config.intra_threads > 0 {
        builder = builder.with_intra_threads(config.intra_threads)?;
    }

    // Set optimization level
    let opt_level = match config.optimization_level {
        GraphOptimizationLevel::Level1 => GraphOptimizationLevel::Level1,
        GraphOptimizationLevel::Level2 => GraphOptimizationLevel::Level2,
        GraphOptimizationLevel::Level3 => GraphOptimizationLevel::Level3,
        GraphOptimizationLevel::Disable => GraphOptimizationLevel::Disable,
    };
    builder = builder.with_optimization_level(opt_level)?;

    Ok(builder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_singleton() {
        let env1 = get_env();
        let env2 = get_env();
        assert!(Arc::ptr_eq(&env1, &env2));
    }

    #[test]
    fn test_session_builder_config() {
        let config = RuntimeConfig {
            inter_threads: 2,
            intra_threads: 2,
            optimization_level: GraphOptimizationLevel::Level1,
        };
        let builder = create_session_builder(&config);
        assert!(builder.is_ok());
    }
} 