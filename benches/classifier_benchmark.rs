use criterion::{black_box, criterion_group, criterion_main, Criterion};
use text_classifier::{Classifier, BuiltinModel, RuntimeConfig};
use ort::GraphOptimizationLevel;

fn setup_benchmark_classifier() -> Classifier {
    Classifier::builder()
        .with_model(BuiltinModel::MiniLM)
        .unwrap()
        .add_class("test", vec!["sample text"])
        .unwrap()
        .build()
        .unwrap()
}

fn bench_tokenization(c: &mut Criterion) {
    let classifier = setup_benchmark_classifier();
    let mut group = c.benchmark_group("Tokenization");
    
    // Configure sampling
    group.sample_size(50);
    group.warm_up_time(std::time::Duration::from_secs(1));
    
    // Short text (< 10 tokens)
    group.bench_function("short_text", |b| b.iter(|| {
        classifier.count_tokens(black_box("This is a short text")).unwrap()
    }));
    
    // Medium text (~50 tokens)
    group.bench_function("medium_text", |b| b.iter(|| {
        classifier.count_tokens(black_box(
            "This is a medium length text that should take more time to tokenize \
             and process due to its increased length and complexity. It contains \
             multiple sentences with various words and punctuation."
        )).unwrap()
    }));
    
    // Long text (~200 tokens)
    group.bench_function("long_text", |b| b.iter(|| {
        classifier.count_tokens(black_box(
            "This is a much longer text that contains multiple paragraphs and should \
             take significantly more time to process. It includes various words, \
             punctuation marks, and different types of sentences.\n\n\
             The second paragraph adds more content and complexity to the text, \
             making it a good test case for tokenization performance with longer \
             documents. We want to ensure the tokenizer handles such cases efficiently.\n\n\
             Finally, this last paragraph helps us understand how the tokenizer scales \
             with input size and whether it maintains good performance even with \
             substantial amounts of text that might be encountered in real-world scenarios."
        )).unwrap()
    }));
    
    group.finish();
}

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Prediction");
    group.sample_size(50);
    group.warm_up_time(std::time::Duration::from_secs(1));
    
    // Test different runtime configurations
    let configs = vec![
        ("single_thread", RuntimeConfig {
            inter_threads: 1,
            intra_threads: 1,
            optimization_level: GraphOptimizationLevel::Level1,
        }),
        ("multi_thread", RuntimeConfig {
            inter_threads: 2,
            intra_threads: 2,
            optimization_level: GraphOptimizationLevel::Level2,
        }),
        ("optimized", RuntimeConfig {
            inter_threads: 0,  // Let ONNX Runtime decide
            intra_threads: 0,  // Let ONNX Runtime decide
            optimization_level: GraphOptimizationLevel::Level3,
        }),
    ];
    
    for (name, config) in configs {
        let classifier = Classifier::builder()
            .with_runtime_config(config)
            .with_model(BuiltinModel::MiniLM)
            .unwrap()
            .add_class("positive", vec!["great", "excellent", "amazing"])
            .unwrap()
            .add_class("negative", vec!["bad", "terrible", "awful"])
            .unwrap()
            .add_class("neutral", vec!["okay", "fine", "average"])
            .unwrap()
            .build()
            .unwrap();
            
        group.bench_function(format!("predict_{}", name), |b| b.iter(|| {
            classifier.predict(black_box(
                "This is a test sentence that needs to be classified into one of the categories."
            )).unwrap()
        }));
    }
    
    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling");
    group.sample_size(50);
    group.warm_up_time(std::time::Duration::from_secs(1));
    
    // Test scaling with number of classes
    let class_counts = [2, 5, 10, 20, 50];
    for &count in &class_counts {
        let mut builder = Classifier::builder()
            .with_model(BuiltinModel::MiniLM)
            .unwrap();
            
        for i in 0..count {
            builder = builder
                .add_class(
                    &format!("class_{}", i),
                    vec!["example 1", "example 2", "example 3"]
                )
                .unwrap();
        }
        
        let classifier = builder.build().unwrap();
        
        group.bench_function(format!("classes_{}", count), |b| b.iter(|| {
            classifier.predict(black_box("Test text for scaling benchmark")).unwrap()
        }));
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tokenization,
    bench_prediction,
    bench_scaling
);
criterion_main!(benches); 