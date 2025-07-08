"""
TensorFlow GPU 메모리 집약적 테스트
이 테스트는 GPU 메모리 사용량을 최대화하여 GPU 메모리 리소스 모니터링을 테스트합니다.
"""

import tensorflow as tf
import numpy as np
import time
import argparse
import sys
import os

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


def setup_tensorflow_gpu():
    """TensorFlow GPU 설정"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU 메모리 성장 허용
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 설정 완료: {len(gpus)}개 GPU 사용 가능")
            return True
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
            return False
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        return False


def allocate_large_gpu_tensors(num_tensors=50, tensor_shape=(4000, 4000)):
    """대량의 GPU 텐서 할당"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return []
    
    print(f"GPU에 대용량 텐서 할당: {num_tensors}개, 크기: {tensor_shape}")
    
    with tf.device('/GPU:0'):
        tensors = []
        for i in range(num_tensors):
            try:
                # 다양한 데이터 타입으로 GPU 텐서 생성
                if i % 3 == 0:
                    tensor = tf.random.normal(tensor_shape, dtype=tf.float32)
                elif i % 3 == 1:
                    tensor = tf.random.normal(tensor_shape, dtype=tf.float16)
                else:
                    tensor = tf.random.uniform(tensor_shape, 0, 100, dtype=tf.int32)
                
                tensors.append(tensor)
                
                if i % 10 == 0:
                    print(f"  할당됨: {i+1}/{num_tensors}")
                    
            except tf.errors.ResourceExhaustedError:
                print(f"  GPU 메모리 부족으로 {i}번째 텐서에서 중단")
                break
            except Exception as e:
                print(f"  오류 발생: {e}")
                break
    
    return tensors


def create_massive_gpu_model():
    """대용량 GPU 모델 생성"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return None
    
    print("대용량 GPU 모델 생성")
    
    with tf.device('/GPU:0'):
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(8192, activation='relu', input_shape=(20000,)),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(2048, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            # 모델 컴파일
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 더미 입력으로 모델 초기화
            dummy_input = tf.random.normal([1, 20000])
            _ = model(dummy_input)
            
            # 모델 파라미터 수 계산
            total_params = model.count_params()
            print(f"  모델 생성 완료: {total_params:,} 파라미터")
            
            return model
            
        except tf.errors.ResourceExhaustedError:
            print("  GPU 메모리 부족으로 모델 생성 실패")
            return None
        except Exception as e:
            print(f"  모델 생성 중 오류: {e}")
            return None


def progressive_gpu_memory_allocation(max_iterations=100):
    """점진적 GPU 메모리 할당"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return []
    
    print(f"점진적 GPU 메모리 할당: {max_iterations} 반복")
    
    with tf.device('/GPU:0'):
        current_size = 1000
        allocated_tensors = []
        
        for i in range(max_iterations):
            try:
                tensor_size = int(current_size)
                tensor = tf.random.normal([tensor_size, tensor_size], dtype=tf.float32)
                allocated_tensors.append(tensor)
                
                current_size *= 1.05
                
                if i % 10 == 0:
                    print(f"  반복 {i}: 크기 {tensor_size}x{tensor_size}")
                    
            except tf.errors.ResourceExhaustedError:
                print(f"  GPU 메모리 한계에 도달: 반복 {i}")
                break
            except Exception as e:
                print(f"  오류 발생: {e}")
                break
        
        return allocated_tensors


def gpu_memory_stress_test(stress_level=3):
    """GPU 메모리 스트레스 테스트"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return
    
    print(f"GPU 메모리 스트레스 테스트 (레벨: {stress_level})")
    
    with tf.device('/GPU:0'):
        base_size = 2000
        tensor_size = base_size * stress_level
        
        stress_tensors = []
        
        for i in range(20):
            try:
                # 다양한 연산으로 GPU 메모리 사용
                a = tf.random.normal([tensor_size, tensor_size])
                b = tf.random.normal([tensor_size, tensor_size])
                
                # 행렬 곱셈
                c = tf.matmul(a, b)
                
                # 추가 연산
                d = tf.sin(c) + tf.cos(c)
                e = tf.matmul(d, tf.transpose(a))
                
                stress_tensors.append(e)
                
                if i % 5 == 0:
                    print(f"  스트레스 {i}: 크기 {tensor_size}x{tensor_size}")
                    
            except tf.errors.ResourceExhaustedError:
                print(f"  스트레스 테스트 중 메모리 부족: 단계 {i}")
                break
            except Exception as e:
                print(f"  오류 발생: {e}")
                break
        
        return stress_tensors


def gpu_training_memory_test(batch_size=128, num_epochs=20):
    """GPU 학습 메모리 테스트"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return
    
    print(f"GPU 학습 메모리 테스트 (배치: {batch_size}, 에포크: {num_epochs})")
    
    with tf.device('/GPU:0'):
        # 메모리 집약적 모델
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4096, activation='relu', input_shape=(5000,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(100, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 학습 루프
        for epoch in range(num_epochs):
            try:
                for batch in range(10):  # 에포크당 10 배치
                    # 배치 데이터 생성
                    X = tf.random.normal([batch_size, 5000])
                    y = tf.random.uniform([batch_size], 0, 100, dtype=tf.int32)
                    
                    # 학습 스텝
                    with tf.GradientTape() as tape:
                        predictions = model(X, training=True)
                        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                        loss = tf.reduce_mean(loss)
                    
                    gradients = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                if epoch % 5 == 0:
                    print(f"  에포크 {epoch}: 손실 {loss.numpy():.4f}")
                    
            except tf.errors.ResourceExhaustedError:
                print(f"  GPU 메모리 부족으로 에포크 {epoch}에서 중단")
                break


def main():
    parser = argparse.ArgumentParser(description='TensorFlow GPU 메모리 집약적 테스트')
    parser.add_argument('--duration', type=int, default=180, help='테스트 지속 시간 (초)')
    parser.add_argument('--num-tensors', type=int, default=30, help='할당할 GPU 텐서 수')
    parser.add_argument('--tensor-size', type=int, default=3000, help='텐서 크기')
    parser.add_argument('--stress-level', type=int, default=2, help='스트레스 테스트 레벨')
    parser.add_argument('--skip-progressive', action='store_true', help='점진적 할당 건너뛰기')
    parser.add_argument('--skip-stress', action='store_true', help='스트레스 테스트 건너뛰기')
    parser.add_argument('--skip-training', action='store_true', help='학습 테스트 건너뛰기')
    
    args = parser.parse_args()
    
    # TensorFlow GPU 설정
    gpu_available = setup_tensorflow_gpu()
    
    # 시스템 정보 출력
    print_system_info()
    
    # TensorFlow 정보
    print(f"\nTensorFlow 버전: {tf.__version__}")
    if gpu_available:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"사용 가능한 GPU: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print(f"\n🎯 TensorFlow GPU 메모리 집약적 테스트 시작 (지속 시간: {args.duration}초)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 테스트 실행
        
        # 1. 대량 GPU 텐서 할당
        print("\n1. 대량 GPU 텐서 할당")
        gpu_tensors = allocate_large_gpu_tensors(
            num_tensors=args.num_tensors,
            tensor_shape=(args.tensor_size, args.tensor_size)
        )
        
        # 2. 대용량 GPU 모델 생성
        print("\n2. 대용량 GPU 모델 생성")
        gpu_model = create_massive_gpu_model()
        
        # 3. 점진적 GPU 메모리 할당
        if not args.skip_progressive:
            print("\n3. 점진적 GPU 메모리 할당")
            progressive_tensors = progressive_gpu_memory_allocation(max_iterations=50)
        
        # 4. GPU 메모리 스트레스 테스트
        if not args.skip_stress:
            print("\n4. GPU 메모리 스트레스 테스트")
            stress_tensors = gpu_memory_stress_test(stress_level=args.stress_level)
        
        # 5. GPU 학습 메모리 테스트
        if not args.skip_training:
            print("\n5. GPU 학습 메모리 테스트")
            gpu_training_memory_test(batch_size=64, num_epochs=15)
        
        # 남은 시간 동안 추가 GPU 텐서 할당
        elapsed_time = time.time() - start_time
        remaining_time = args.duration - elapsed_time
        
        if remaining_time > 15:
            print(f"\n6. 추가 GPU 텐서 할당 ({remaining_time:.1f}초 동안)")
            extra_tensors = allocate_large_gpu_tensors(
                num_tensors=max(5, int(remaining_time / 10)),
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
        
        # 최종 리소스 사용량 확인
        final_usage = monitor.get_current_usage()
        print(f"\n최종 리소스 사용량:")
        if gpu_available:
            for i in range(len(tf.config.list_physical_devices('GPU'))):
                gpu_memory = final_usage.get(f'gpu_{i}_memory_used', 0)
                print(f"GPU {i} 메모리: {gpu_memory:.1f} MB")
        
        print("\n🎯 TensorFlow GPU 메모리 집약적 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    main() 