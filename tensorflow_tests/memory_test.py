"""
TensorFlow 메모리 집약적 테스트
이 테스트는 메모리 사용량을 최대화하여 메모리 리소스 모니터링을 테스트합니다.
"""

import tensorflow as tf
import numpy as np
import time
import argparse
import sys
import os
import gc

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


class TensorFlowMemoryConsumer:
    """TensorFlow 메모리 소비자 클래스"""
    def __init__(self):
        self.tensor_storage = []
        self.model_storage = []
        
        # CPU만 사용하도록 설정
        tf.config.set_visible_devices([], 'GPU')
        print("TensorFlow 메모리 테스트 - CPU 모드로 설정됨")
    
    def allocate_large_tensors(self, num_tensors=100, tensor_shape=(5000, 5000)):
        """대량의 텐서를 메모리에 할당"""
        print(f"대용량 텐서 할당: {num_tensors}개, 크기: {tensor_shape}")
        
        tensors = []
        for i in range(num_tensors):
            # 다양한 데이터 타입으로 텐서 생성
            if i % 4 == 0:
                tensor = tf.random.normal(tensor_shape, dtype=tf.float32)
            elif i % 4 == 1:
                tensor = tf.random.normal(tensor_shape, dtype=tf.float64)
            elif i % 4 == 2:
                tensor = tf.random.uniform(tensor_shape, 0, 100, dtype=tf.int32)
            else:
                tensor = tf.ones(tensor_shape, dtype=tf.float32)
            
            tensors.append(tensor)
            
            if i % 10 == 0:
                print(f"  할당됨: {i+1}/{num_tensors}")
        
        self.tensor_storage.extend(tensors)
        return tensors
    
    def create_memory_intensive_models(self, num_models=5):
        """메모리 집약적 모델들 생성"""
        print(f"메모리 집약적 모델 {num_models}개 생성")
        
        models = []
        for i in range(num_models):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(4096, activation='relu', input_shape=(10000,)),
                tf.keras.layers.Dense(2048, activation='relu'),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            # 모델 컴파일
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            
            # 더미 데이터로 모델 초기화
            dummy_input = tf.random.normal([1, 10000])
            _ = model(dummy_input)
            
            models.append(model)
            print(f"  모델 {i+1} 생성 완료")
        
        self.model_storage.extend(models)
        return models
    
    def generate_large_datasets(self, num_samples=100000, feature_size=5000):
        """대용량 데이터셋 생성"""
        print(f"대용량 데이터셋 생성: {num_samples} 샘플 x {feature_size} 특성")
        
        # 기본 데이터
        X = tf.random.normal([num_samples, feature_size])
        y = tf.random.uniform([num_samples], 0, 1000, dtype=tf.int32)
        
        # 데이터 변환으로 메모리 사용량 증가
        X_normalized = tf.nn.l2_normalize(X, axis=1)
        X_squared = tf.square(X)
        X_log = tf.math.log(tf.abs(X) + 1e-8)
        X_combined = tf.concat([X, X_normalized, X_squared, X_log], axis=1)
        
        dataset = {
            'X': X,
            'y': y,
            'X_normalized': X_normalized,
            'X_squared': X_squared,
            'X_log': X_log,
            'X_combined': X_combined
        }
        
        return dataset
    
    def progressive_memory_allocation(self, max_iterations=100):
        """점진적 메모리 할당"""
        print(f"점진적 메모리 할당: {max_iterations} 반복")
        
        current_size = 1000
        allocated_tensors = []
        
        for i in range(max_iterations):
            try:
                tensor_size = int(current_size)
                tensor = tf.random.normal([tensor_size, tensor_size])
                allocated_tensors.append(tensor)
                
                current_size *= 1.1
                
                if i % 10 == 0:
                    print(f"  반복 {i}: 크기 {tensor_size}x{tensor_size}")
                    
            except Exception as e:
                print(f"  메모리 한계 도달: 반복 {i}")
                break
        
        return allocated_tensors
    
    def memory_leak_simulation(self, iterations=50):
        """메모리 누수 시뮬레이션"""
        print(f"메모리 누수 시뮬레이션: {iterations} 반복")
        
        leaked_tensors = []
        
        for i in range(iterations):
            # 대용량 텐서 생성
            tensor = tf.random.normal([3000, 3000])
            
            # 복잡한 연산 수행
            result = tf.matmul(tensor, tf.transpose(tensor))
            result = tf.sin(result) + tf.cos(result)
            result = tf.exp(tf.clip_by_value(result, -3, 3))
            
            # 결과 저장 (누수 시뮬레이션)
            leaked_tensors.append(result)
            
            if i % 10 == 0:
                print(f"  누수 시뮬레이션 {i}/{iterations}")
        
        return leaked_tensors
    
    def clear_memory(self):
        """메모리 정리"""
        print("메모리 정리 중...")
        self.tensor_storage.clear()
        self.model_storage.clear()
        gc.collect()


def batch_processing_memory_test(batch_size=1000, num_batches=100):
    """배치 처리 메모리 테스트"""
    print(f"배치 처리 메모리 테스트: {num_batches} 배치, 크기: {batch_size}")
    
    # 메모리 집약적 모델
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation='relu', input_shape=(3000,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(100, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    for batch_idx in range(num_batches):
        # 배치 데이터 생성
        X = tf.random.normal([batch_size, 3000])
        y = tf.random.uniform([batch_size], 0, 100, dtype=tf.int32)
        
        # 학습 스텝
        with tf.GradientTape() as tape:
            predictions = model(X, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if batch_idx % 20 == 0:
            print(f"  배치 {batch_idx}/{num_batches}")


def main():
    parser = argparse.ArgumentParser(description='TensorFlow 메모리 집약적 테스트')
    parser.add_argument('--duration', type=int, default=120, help='테스트 지속 시간 (초)')
    parser.add_argument('--num-tensors', type=int, default=50, help='할당할 텐서 수')
    parser.add_argument('--tensor-size', type=int, default=3000, help='텐서 크기')
    parser.add_argument('--num-models', type=int, default=3, help='생성할 모델 수')
    parser.add_argument('--skip-progressive', action='store_true', help='점진적 할당 건너뛰기')
    parser.add_argument('--skip-leak', action='store_true', help='메모리 누수 테스트 건너뛰기')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # TensorFlow 정보
    print(f"\nTensorFlow 버전: {tf.__version__}")
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print(f"\n🧠 TensorFlow 메모리 집약적 테스트 시작 (지속 시간: {args.duration}초)")
    print("=" * 60)
    
    memory_consumer = TensorFlowMemoryConsumer()
    start_time = time.time()
    
    try:
        # 테스트 실행
        
        # 1. 대량 텐서 할당
        print("\n1. 대량 텐서 할당")
        tensors = memory_consumer.allocate_large_tensors(
            num_tensors=args.num_tensors,
            tensor_shape=(args.tensor_size, args.tensor_size)
        )
        
        # 2. 메모리 집약적 모델 생성
        print("\n2. 메모리 집약적 모델 생성")
        models = memory_consumer.create_memory_intensive_models(num_models=args.num_models)
        
        # 3. 대용량 데이터셋 생성
        print("\n3. 대용량 데이터셋 생성")
        dataset = memory_consumer.generate_large_datasets(num_samples=30000, feature_size=3000)
        
        # 4. 점진적 메모리 할당
        if not args.skip_progressive:
            print("\n4. 점진적 메모리 할당")
            progressive_tensors = memory_consumer.progressive_memory_allocation(max_iterations=50)
        
        # 5. 메모리 누수 시뮬레이션
        if not args.skip_leak:
            print("\n5. 메모리 누수 시뮬레이션")
            leaked_tensors = memory_consumer.memory_leak_simulation(iterations=30)
        
        # 6. 배치 처리 테스트
        print("\n6. 배치 처리 테스트")
        batch_processing_memory_test(batch_size=500, num_batches=50)
        
        # 남은 시간 동안 추가 텐서 할당
        elapsed_time = time.time() - start_time
        remaining_time = args.duration - elapsed_time
        
        if remaining_time > 10:
            print(f"\n7. 추가 텐서 할당 ({remaining_time:.1f}초 동안)")
            extra_tensors = memory_consumer.allocate_large_tensors(
                num_tensors=max(10, int(remaining_time / 5)),
                tensor_shape=(2000, 2000)
            )
        
    except KeyboardInterrupt:
        print("\n테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 리소스 모니터링 종료
        monitor.stop_monitoring()
        monitor.print_summary()
        
        # 최종 리소스 사용량
        final_usage = monitor.get_current_usage()
        print(f"\n최종 리소스 사용량:")
        print(f"Memory: {final_usage.get('memory_percent', 0):.1f}%")
        print(f"Memory MB: {final_usage.get('memory_mb', 0):.1f} MB")
        
        # 메모리 정리
        memory_consumer.clear_memory()
        
        print("\n🎯 TensorFlow 메모리 집약적 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    main() 