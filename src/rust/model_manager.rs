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

use crate::models::{BuiltinModel, ModelInfo};

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

    pub fn get_model_path(&self, model: BuiltinModel) -> PathBuf {
        let info = model.get_model_info();
        self.models_dir.join(info.name).join("model.onnx")
    }

    pub fn get_tokenizer_path(&self, model: BuiltinModel) -> PathBuf {
        let info = model.get_model_info();
        self.models_dir.join(info.name).join("tokenizer.json")
    }

    pub fn is_model_downloaded(&self, model: BuiltinModel) -> bool {
        let model_path = self.get_model_path(model);
        let tokenizer_path = self.get_tokenizer_path(model);
        log::info!("Checking if model is downloaded:");
        log::info!("  Model path: {:?} (exists: {})", model_path, model_path.exists());
        log::info!("  Tokenizer path: {:?} (exists: {})", tokenizer_path, tokenizer_path.exists());
        model_path.exists() && tokenizer_path.exists()
    }

    pub async fn download_model(&self, model: BuiltinModel) -> Result<(), ModelError> {
        let info = model.get_model_info();
        let _lock = self.download_lock.lock().await;
        
        // Create directory
        let model_dir = self.models_dir.join(&info.name);
        log::info!("Creating model directory at {:?}", model_dir);
        fs::create_dir_all(&model_dir)?;

        // Handle model file
        let model_path = self.get_model_path(model);
        log::info!("Model path: {:?}", model_path);
        let model_result = if model_path.exists() {
            log::info!("Model file exists at {:?}, verifying...", model_path);
            if !self.verify_file(&model_path, &info.model_hash)? {
                log::warn!("Model file verification failed, redownloading");
                self.download_and_verify_model(&info, &model_path).await
            } else {
                log::info!("Existing model file verified successfully");
                Ok(())
            }
        } else {
            log::info!("Model file does not exist, downloading...");
            self.download_and_verify_model(&info, &model_path).await
        };

        // Handle tokenizer file
        let tokenizer_path = self.get_tokenizer_path(model);
        log::info!("Tokenizer path: {:?}", tokenizer_path);
        let tokenizer_result = if tokenizer_path.exists() {
            log::info!("Tokenizer file exists at {:?}, verifying...", tokenizer_path);
            if !self.verify_file(&tokenizer_path, &info.tokenizer_hash)? {
                log::warn!("Tokenizer file verification failed, redownloading");
                self.download_and_verify_tokenizer(&info, &tokenizer_path).await
            } else {
                log::info!("Existing tokenizer file verified successfully");
                Ok(())
            }
        } else {
            log::info!("Tokenizer file does not exist, downloading...");
            self.download_and_verify_tokenizer(&info, &tokenizer_path).await
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
                let _ = self.remove_download(model);
                Err(e)
            }
            (_, Err(e)) => {
                log::error!("Failed to setup tokenizer file: {}", e);
                // Cleanup on failure
                let _ = self.remove_download(model);
                Err(e)
            }
        }
    }

    fn verify_file(&self, path: &Path, expected_hash: &str) -> Result<bool, ModelError> {
        log::info!("Verifying file: {:?}", path);
        let bytes = fs::read(path)?;
        log::info!("Read {} bytes", bytes.len());
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());
        log::info!("Calculated hash: {}", hash);
        log::info!("Expected hash:   {}", expected_hash);
        Ok(hash == expected_hash)
    }

    pub fn verify_model(&self, model: BuiltinModel) -> Result<bool, ModelError> {
        let info = model.get_model_info();
        let model_path = self.get_model_path(model);
        let tokenizer_path = self.get_tokenizer_path(model);

        log::info!("Verifying model files:");
        log::info!("  Model path: {:?}", model_path);
        log::info!("  Tokenizer path: {:?}", tokenizer_path);

        if !model_path.exists() || !tokenizer_path.exists() {
            log::info!("One or both files do not exist");
            return Ok(false);
        }

        let model_ok = self.verify_file(&model_path, &info.model_hash)?;
        let tokenizer_ok = self.verify_file(&tokenizer_path, &info.tokenizer_hash)?;

        log::info!("Verification results:");
        log::info!("  Model hash verification: {}", model_ok);
        log::info!("  Tokenizer hash verification: {}", tokenizer_ok);

        Ok(model_ok && tokenizer_ok)
    }

    async fn download_and_verify_file(
        &self,
        url: &str,
        path: &Path,
        expected_hash: &str,
        file_type: &str,
    ) -> Result<(), ModelError> {
        log::info!("Downloading {} file from {} to {:?}", file_type, url, path);
        let response = reqwest::get(url).await?;
        log::info!("Download response status: {}", response.status());
        let bytes = response.bytes().await?;
        log::info!("Downloaded {} bytes", bytes.len());
        
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());
        log::info!("Calculated hash: {}", hash);
        
        if hash != expected_hash {
            log::error!("{} hash mismatch: expected {}, got {}", file_type, expected_hash, hash);
            return Err(ModelError::HashMismatch {
                file_type: file_type.to_string(),
                expected: expected_hash.to_string(),
                actual: hash,
            });
        }
        
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            log::info!("Creating parent directory: {:?}", parent);
            fs::create_dir_all(parent)?;
        }
        
        log::info!("Writing {} bytes to {:?}", bytes.len(), path);
        fs::write(path, bytes)?;
        
        // Verify after writing
        log::info!("Verifying written file");
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

    pub fn remove_download(&self, model: BuiltinModel) -> Result<(), ModelError> {
        let model_path = self.get_model_path(model);
        let tokenizer_path = self.get_tokenizer_path(model);
        
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
    pub async fn ensure_model_downloaded(&self, model: BuiltinModel) -> Result<(), ModelError> {
        log::info!("Checking if model {:?} is downloaded...", model);
        if !self.is_model_downloaded(model) {
            log::info!("Model not found, downloading...");
            self.download_model(model).await?;
        } else {
            log::info!("Model exists, verifying...");
            if !self.verify_model(model)? {
                log::info!("Model verification failed, re-downloading...");
                self.remove_download(model)?;
                self.download_model(model).await?;
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
        let model = BuiltinModel::MiniLM;

        // Clean up any existing files
        let model_path = manager.get_model_path(model);
        let tokenizer_path = manager.get_tokenizer_path(model);
        if model_path.exists() {
            std::fs::remove_file(&model_path)?;
        }
        if tokenizer_path.exists() {
            std::fs::remove_file(&tokenizer_path)?;
        }

        assert!(!manager.is_model_downloaded(model));
        
        let result = manager.download_model(model).await;
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