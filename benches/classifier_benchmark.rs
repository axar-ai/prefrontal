use criterion::{black_box, criterion_group, criterion_main, Criterion};
use text_classifier::Classifier;

fn setup_benchmark_classifier() -> Classifier {
    Classifier::new(
        "models/onnx-minilm/model.onnx",
        "models/onnx-minilm/tokenizer.json"
    )
}

fn bench_tokenization(c: &mut Criterion) {
    let classifier = setup_benchmark_classifier();
    let mut group = c.benchmark_group("Tokenization");
    
    // Short text
    group.bench_function("short_text", |b| b.iter(|| {
        classifier.tokenize(black_box("This is a short text"))
    }));
    
    // Medium text
    group.bench_function("medium_text", |b| b.iter(|| {
        classifier.tokenize(black_box(
            "This is a medium length text that should take more time to tokenize \
             and process due to its increased length and complexity"
        ))
    }));
    
    // Long text
    group.bench_function("long_text", |b| b.iter(|| {
        classifier.tokenize(black_box(
            "This is a much longer text that contains multiple sentences and should \
             take significantly more time to process. It includes various words and \
             punctuation marks, making it a good test case for tokenization performance. \
             The length of this text should help us understand how the tokenizer scales \
             with input size."
        ))
    }));
    
    group.finish();
}

fn bench_embedding_generation(c: &mut Criterion) {
    let classifier = setup_benchmark_classifier();
    let mut group = c.benchmark_group("Embedding");
    
    group.bench_function("single_text", |b| b.iter(|| {
        classifier.embed_text(black_box("Sample text for embedding benchmark"))
    }));
    
    group.finish();
}

fn bench_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("Classification");
    
    // Single class classification with timing
    {
        let mut classifier = setup_benchmark_classifier();
        classifier.add_class("sports", vec!["football game", "basketball match"]);
        classifier.build().unwrap();
        
        group.bench_function("single_prediction", |b| b.iter(|| {
            classifier.predict(black_box("Test text for benchmark"))
        }));
    }
    
    // Multiple classes with timing
    {
        let mut classifier = setup_benchmark_classifier();
        for i in 0..10 {
            classifier.add_class(
                &format!("class_{}", i),
                vec!["example 1", "example 2", "example 3"]
            );
        }
        classifier.build().unwrap();
        
        group.bench_function("prediction_ten_classes", |b| b.iter(|| {
            classifier.predict(black_box("Test text for benchmark"))
        }));
    }
    
    group.finish();
}

fn bench_build_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("Build");
    
    group.bench_function("build_ten_classes", |b| b.iter(|| {
        let mut classifier = setup_benchmark_classifier();
        for i in 0..10 {
            classifier.add_class(
                &format!("class_{}", i),
                vec!["example 1", "example 2", "example 3"]
            );
        }
        classifier.build().unwrap()
    }));
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tokenization,
    bench_embedding_generation,
    bench_classification,
    bench_build_time
);
criterion_main!(benches); 