use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::Result as OrtResult;
use std::sync::Once;

static INIT: Once = Once::new();

#[derive(Debug)]
pub struct RuntimeConfig {
    /// Number of threads to use for parallel model execution (0 = let ONNX Runtime decide)
    /// 
    /// This controls inter-op parallelism, which is parallelism between independent nodes in the model graph.
    /// Setting this to 0 allows ONNX Runtime to automatically choose based on system resources.
    pub inter_threads: usize,

    /// Number of threads to use for parallel computation within nodes (0 = let ONNX Runtime decide)
    /// 
    /// This controls intra-op parallelism, which is parallelism within individual operations like matrix multiplication.
    /// Setting this to 0 allows ONNX Runtime to automatically choose based on system resources.
    pub intra_threads: usize,

    /// The level of graph optimization to apply
    /// 
    /// Higher levels perform more aggressive optimizations but may take longer to compile.
    /// Level3 (maximum) is recommended for production use.
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

fn init_onnx_environment() -> OrtResult<()> {
    ort::init()
        .with_name("prefrontal")
        .commit()?;
    Ok(())
}

pub fn ensure_initialized() -> OrtResult<()> {
    INIT.call_once(|| {
        init_onnx_environment().expect("Failed to initialize ONNX Runtime environment");
    });
    Ok(())
}

pub fn create_session_builder(config: &RuntimeConfig) -> OrtResult<SessionBuilder> {
    ensure_initialized()?;
    let mut builder = Session::builder()?;

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
    fn test_environment_initialization() {
        assert!(ensure_initialized().is_ok());
        assert!(ensure_initialized().is_ok()); // Second call should be fine
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

    #[test]
    fn test_default_config() {
        let config = RuntimeConfig::default();
        assert_eq!(config.inter_threads, 0);
        assert_eq!(config.intra_threads, 0);
        match config.optimization_level {
            GraphOptimizationLevel::Level3 => (),
            _ => panic!("Expected GraphOptimizationLevel::Level3"),
        }
    }
} 