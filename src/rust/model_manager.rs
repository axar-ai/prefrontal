use std::path::{Path, PathBuf};
use std::fs;
use std::io;
use std::sync::Arc;
use std::env;
use tokio::sync::Mutex;
use reqwest;
use sha2::{Sha256, Digest};
use dirs;
use log;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not downloaded: {0}")]
    NotDownloaded(String),
    #[error("Download error: {0}")]
    DownloadError(#[from] reqwest::Error),
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    #[error("Model verification failed")]
    VerificationFailed,
    #[error("Hash mismatch: expected {expected}, got {actual} for {file_type} file")]
    HashMismatch {
        file_type: String,
        expected: String,
        actual: String,
    },
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub model_url: String,
    pub tokenizer_url: String,
    pub model_hash: String,
    pub tokenizer_hash: String,
}

#[derive(Clone)]
pub struct ModelManager {
    models_dir: PathBuf,
    download_lock: Arc<Mutex<()>>,
}

impl ModelManager {
    /// Creates a new ModelManager with the default models directory
    pub fn new_default() -> io::Result<Self> {
        Ok(Self::new(Self::get_default_models_dir())?)
    }

    /// Returns the default models directory path
    pub fn get_default_models_dir() -> PathBuf {
        // 1. Check environment variable
        if let Ok(path) = env::var("PREFRONTAL_CACHE") {
            return PathBuf::from(path).join("models");
        }

        // 2. Use platform-specific cache directory
        if let Some(cache_dir) = dirs::cache_dir() {
            return cache_dir.join("prefrontal").join("models");
        }

        // 3. Fallback to user's home directory
        if let Some(home_dir) = dirs::home_dir() {
            return home_dir.join(".cache").join("prefrontal").join("models");
        }

        // 4. If all else fails, use system temp directory (platform agnostic)
        env::temp_dir().join("prefrontal").join("models")
    }

    pub fn new<P: AsRef<Path>>(models_dir: P) -> io::Result<Self> {
        let models_dir = models_dir.as_ref().to_path_buf();
        fs::create_dir_all(&models_dir)?;
        Ok(Self {
            models_dir,
            download_lock: Arc::new(Mutex::new(())),
        })
    }

    pub fn get_model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name).join("model.onnx")
    }

    pub fn get_tokenizer_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(model_name).join("tokenizer.json")
    }

    pub fn is_model_downloaded(&self, model_name: &str) -> bool {
        let model_path = self.get_model_path(model_name);
        let tokenizer_path = self.get_tokenizer_path(model_name);
        model_path.exists() && tokenizer_path.exists()
    }

    pub async fn download_model(&self, info: &ModelInfo) -> Result<(), ModelError> {
        let _lock = self.download_lock.lock().await;
        
        // Create directory
        let model_dir = self.models_dir.join(&info.name);
        fs::create_dir_all(&model_dir)?;

        // Handle model file
        let model_path = self.get_model_path(&info.name);
        let model_result = if model_path.exists() {
            log::info!("Model file exists at {:?}, verifying...", model_path);
            if !self.verify_file(&model_path, &info.model_hash)? {
                log::warn!("Model file verification failed, redownloading");
                self.download_and_verify_model(info, &model_path).await
            } else {
                log::info!("Existing model file verified successfully");
                Ok(())
            }
        } else {
            self.download_and_verify_model(info, &model_path).await
        };

        // Handle tokenizer file
        let tokenizer_path = self.get_tokenizer_path(&info.name);
        let tokenizer_result = if tokenizer_path.exists() {
            log::info!("Tokenizer file exists at {:?}, verifying...", tokenizer_path);
            if !self.verify_file(&tokenizer_path, &info.tokenizer_hash)? {
                log::warn!("Tokenizer file verification failed, redownloading");
                self.download_and_verify_tokenizer(info, &tokenizer_path).await
            } else {
                log::info!("Existing tokenizer file verified successfully");
                Ok(())
            }
        } else {
            self.download_and_verify_tokenizer(info, &tokenizer_path).await
        };

        // Handle results
        match (model_result, tokenizer_result) {
            (Ok(()), Ok(())) => {
                log::info!("Model and tokenizer ready to use");
                Ok(())
            }
            (Err(e), _) => {
                log::error!("Failed to setup model file: {}", e);
                // Cleanup on failure
                let _ = self.remove_download(&info.name);
                Err(e)
            }
            (_, Err(e)) => {
                log::error!("Failed to setup tokenizer file: {}", e);
                // Cleanup on failure
                let _ = self.remove_download(&info.name);
                Err(e)
            }
        }
    }

    fn verify_file(&self, path: &Path, expected_hash: &str) -> Result<bool, ModelError> {
        let bytes = fs::read(path)?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());
        Ok(hash == expected_hash)
    }

    pub fn verify_model(&self, info: &ModelInfo) -> Result<bool, ModelError> {
        let model_path = self.get_model_path(&info.name);
        let tokenizer_path = self.get_tokenizer_path(&info.name);

        if !model_path.exists() || !tokenizer_path.exists() {
            return Ok(false);
        }

        Ok(
            self.verify_file(&model_path, &info.model_hash)? && 
            self.verify_file(&tokenizer_path, &info.tokenizer_hash)?
        )
    }

    async fn download_and_verify_file(
        &self,
        url: &str,
        path: &Path,
        expected_hash: &str,
        file_type: &str,
    ) -> Result<(), ModelError> {
        log::info!("Downloading {} file to {:?}", file_type, path);
        let bytes = reqwest::get(url).await?.bytes().await?;
        
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());
        
        if hash != expected_hash {
            log::error!("{} hash mismatch: expected {}, got {}", file_type, expected_hash, hash);
            return Err(ModelError::HashMismatch {
                file_type: file_type.to_string(),
                expected: expected_hash.to_string(),
                actual: hash,
            });
        }
        
        fs::write(path, bytes)?;
        
        // Verify after writing
        if !self.verify_file(path, expected_hash)? {
            return Err(ModelError::VerificationFailed);
        }
        
        log::info!("{} file downloaded and verified successfully", file_type);
        Ok(())
    }

    async fn download_and_verify_model(&self, info: &ModelInfo, path: &Path) -> Result<(), ModelError> {
        self.download_and_verify_file(&info.model_url, path, &info.model_hash, "model").await
    }

    async fn download_and_verify_tokenizer(&self, info: &ModelInfo, path: &Path) -> Result<(), ModelError> {
        self.download_and_verify_file(&info.tokenizer_url, path, &info.tokenizer_hash, "tokenizer").await
    }

    pub fn remove_download(&self, model_name: &str) -> Result<(), ModelError> {
        let model_path = self.get_model_path(model_name);
        let tokenizer_path = self.get_tokenizer_path(model_name);
        
        if model_path.exists() {
            fs::remove_file(&model_path)?;
        }
        if tokenizer_path.exists() {
            fs::remove_file(&tokenizer_path)?;
        }
        Ok(())
    }

    /// Ensures that a model is downloaded and verified.
    /// If the model doesn't exist, it will be downloaded.
    /// If verification fails, it will be re-downloaded.
    pub async fn ensure_model_downloaded(&self, info: &ModelInfo) -> Result<(), ModelError> {
        log::info!("Checking if model {} is downloaded...", info.name);
        if !self.is_model_downloaded(&info.name) {
            log::info!("Model not found, downloading...");
            self.download_model(info).await?;
        } else {
            log::info!("Model exists, verifying...");
            if !self.verify_model(info)? {
                log::info!("Model verification failed, re-downloading...");
                self.remove_download(&info.name)?;
                self.download_model(info).await?;
            } else {
                log::info!("Model verification successful");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_manager() -> Result<(), ModelError> {
        let manager = ModelManager::new("/tmp/test-cache/models").unwrap();
        let info = ModelInfo {
            name: "minilm".to_string(),
            model_url: "https://huggingface.co/axar-ai/minilm/resolve/main/model.onnx".to_string(),
            tokenizer_url: "https://huggingface.co/axar-ai/minilm/resolve/main/tokenizer.json".to_string(),
            model_hash: "37f1ea074b7166e87295fce31299287d5fb79f76b8b7227fccc8a9f2f1ba4e16".to_string(),
            tokenizer_hash: "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0".to_string(),
        };

        // Clean up any existing files
        let model_path = manager.get_model_path(&info.name);
        let tokenizer_path = manager.get_tokenizer_path(&info.name);
        if model_path.exists() {
            std::fs::remove_file(&model_path)?;
        }
        if tokenizer_path.exists() {
            std::fs::remove_file(&tokenizer_path)?;
        }

        assert!(!manager.is_model_downloaded("minilm"));
        
        let result = manager.download_model(&info).await;
        assert!(result.is_ok());

        Ok(())
    }

    #[test]
    fn test_default_models_dir() {
        // Test with environment variable
        env::set_var("PREFRONTAL_CACHE", "/tmp/test-cache");
        let path = ModelManager::get_default_models_dir();
        assert!(path.to_str().unwrap().contains("/tmp/test-cache/models"));
        env::remove_var("PREFRONTAL_CACHE");

        // Test without environment variable
        let path = ModelManager::get_default_models_dir();
        assert!(path.to_str().unwrap().contains("prefrontal/models"));
    }
} 