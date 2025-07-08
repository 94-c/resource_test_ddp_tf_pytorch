"""
PyTorch GPU 사용률 집약적 테스트
이 테스트는 GPU 사용률을 최대화하여 GPU 활용도 모니터링을 테스트합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import sys
import os
import threading
import concurrent.futures
from typing import List, Dict, Optional

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


class GPUUtilizationMaximizer:
    """GPU 사용률 최대화 클래스"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.streams = []
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA가 사용 불가능합니다. CPU 모드로 실행됩니다.")
        else:
            print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name()}")
            # 여러 CUDA 스트림 생성
            for i in range(8):
                stream = torch.cuda.Stream()
                self.streams.append(stream)
    
    def parallel_matrix_operations(self, num_operations=1000, matrix_size=2000):
        """병렬 행렬 연산으로 GPU 사용률 최대화"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return
        
        print(f"병렬 행렬 연산 시작 ({num_operations} 연산, 크기: {matrix_size})")
        
        def matrix_worker(stream_idx, operations_per_stream):
            """각 스트림에서 실행할 행렬 연산"""
            stream = self.streams[stream_idx % len(self.streams)]
            
            with torch.cuda.stream(stream):
                for i in range(operations_per_stream):
                    # 다양한 행렬 연산
                    a = torch.randn(matrix_size, matrix_size, device=self.device)
                    b = torch.randn(matrix_size, matrix_size, device=self.device)
                    
                    # 행렬 곱셈
                    c = torch.matmul(a, b)
                    
                    # 고유값 분해 (계산 집약적)
                    if i % 10 == 0:
                        try:
                            eigenvalues = torch.linalg.eigvals(c[:500, :500])
                        except:
                            pass
                    
                    # 삼각함수 연산
                    d = torch.sin(c) + torch.cos(c) + torch.tan(torch.clamp(c, -1, 1))
                    
                    # 지수 연산
                    e = torch.exp(torch.clamp(d, -5, 5))
                    
                    # 로그 연산
                    f = torch.log(torch.abs(e) + 1e-8)
                    
                    # 역행렬 계산
                    try:
                        g = torch.linalg.inv(f + torch.eye(matrix_size, device=self.device) * 1e-3)
                    except:
                        g = f
                    
                    # 결과 저장 (메모리 압박 방지를 위해 주기적 정리)
                    if i % 50 == 0:
                        torch.cuda.empty_cache()
        
        # 병렬 실행
        operations_per_stream = num_operations // len(self.streams)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.streams)) as executor:
            futures = []
            for i in range(len(self.streams)):
                future = executor.submit(matrix_worker, i, operations_per_stream)
                futures.append(future)
            
            # 모든 작업 완료 대기
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"스트림 작업 중 오류: {e}")
        
        # 모든 스트림 동기화
        for stream in self.streams:
            stream.synchronize()
    
    def convolution_intensive_test(self, num_iterations=500, batch_size=32):
        """컨볼루션 집약적 테스트"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return
        
        print(f"컨볼루션 집약적 테스트 ({num_iterations} 반복, 배치: {batch_size})")
        
        # 컨볼루션 네트워크 생성
        class ConvIntensiveNet(nn.Module):
            def __init__(self):
                super(ConvIntensiveNet, self).__init__()
                self.conv_layers = nn.ModuleList([
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                    nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.Conv2d(64, 3, kernel_size=3, padding=1)
                ])
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm2d(64),
                    nn.BatchNorm2d(128),
                    nn.BatchNorm2d(256),
                    nn.BatchNorm2d(512),
                    nn.BatchNorm2d(1024),
                    nn.BatchNorm2d(512),
                    nn.BatchNorm2d(256),
                    nn.BatchNorm2d(128),
                    nn.BatchNorm2d(64),
                ])
            
            def forward(self, x):
                for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
                    x = conv(x)
                    x = F.relu(x)
                    x = bn(x)
                    
                    # 다운샘플링과 업샘플링
                    if i < 4:
                        x = F.max_pool2d(x, 2)
                    elif i >= 5:
                        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                
                x = self.conv_layers[-1](x)
                return x
        
        model = ConvIntensiveNet().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for i in range(num_iterations):
            # 랜덤 이미지 데이터 생성
            input_data = torch.randn(batch_size, 3, 256, 256, device=self.device)
            target = torch.randn(batch_size, 3, 256, 256, device=self.device)
            
            # 순전파
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"  컨볼루션 반복 {i}/{num_iterations}, Loss: {loss.item():.4f}")
    
    def transformer_intensive_test(self, num_iterations=200, seq_len=512, batch_size=16):
        """트랜스포머 집약적 테스트"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return
        
        print(f"트랜스포머 집약적 테스트 ({num_iterations} 반복, 시퀀스: {seq_len}, 배치: {batch_size})")
        
        # 트랜스포머 모델 생성
        class IntensiveTransformer(nn.Module):
            def __init__(self, d_model=512, nhead=8, num_layers=6):
                super(IntensiveTransformer, self).__init__()
                self.d_model = d_model
                self.embedding = nn.Embedding(10000, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=2048,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_layer = nn.Linear(d_model, 10000)
            
            def forward(self, x):
                x = self.embedding(x)
                x = x + self.pos_encoding[:x.size(1), :]
                x = self.transformer(x)
                x = self.output_layer(x)
                return x
        
        model = IntensiveTransformer().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for i in range(num_iterations):
            # 랜덤 시퀀스 데이터 생성
            input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=self.device)
            target = torch.randint(0, 10000, (batch_size, seq_len), device=self.device)
            
            # 순전파
            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output.view(-1, 10000), target.view(-1))
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print(f"  트랜스포머 반복 {i}/{num_iterations}, Loss: {loss.item():.4f}")
    
    def fft_intensive_test(self, num_iterations=1000, signal_size=8192):
        """FFT 집약적 테스트"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return
        
        print(f"FFT 집약적 테스트 ({num_iterations} 반복, 신호 크기: {signal_size})")
        
        for i in range(num_iterations):
            # 복잡한 신호 생성
            real_signal = torch.randn(signal_size, device=self.device)
            imag_signal = torch.randn(signal_size, device=self.device)
            complex_signal = torch.complex(real_signal, imag_signal)
            
            # FFT 연산
            fft_result = torch.fft.fft(complex_signal)
            
            # 역 FFT
            ifft_result = torch.fft.ifft(fft_result)
            
            # 2D FFT (이미지 처리 시뮬레이션)
            if i % 10 == 0:
                image_size = int(np.sqrt(signal_size))
                if image_size * image_size == signal_size:
                    image = real_signal.reshape(image_size, image_size)
                    fft_2d = torch.fft.fft2(image)
                    ifft_2d = torch.fft.ifft2(fft_2d)
            
            # 스펙트럼 분석
            power_spectrum = torch.abs(fft_result) ** 2
            
            if i % 100 == 0:
                print(f"  FFT 반복 {i}/{num_iterations}")
    
    def mixed_precision_test(self, num_iterations=300):
        """혼합 정밀도 테스트"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return
        
        print(f"혼합 정밀도 테스트 ({num_iterations} 반복)")
        
        # 혼합 정밀도 모델
        class MixedPrecisionModel(nn.Module):
            def __init__(self):
                super(MixedPrecisionModel, self).__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(2048, 4096),
                    nn.Linear(4096, 2048),
                    nn.Linear(2048, 1024),
                    nn.Linear(1024, 512),
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 64),
                    nn.Linear(64, 10)
                ])
            
            def forward(self, x):
                for layer in self.layers[:-1]:
                    x = F.relu(layer(x))
                return self.layers[-1](x)
        
        model = MixedPrecisionModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        
        for i in range(num_iterations):
            # 배치 데이터 생성
            input_data = torch.randn(128, 2048, device=self.device)
            target = torch.randint(0, 10, (128,), device=self.device)
            
            optimizer.zero_grad()
            
            # 혼합 정밀도 순전파
            with torch.cuda.amp.autocast():
                output = model(input_data)
                loss = criterion(output, target)
            
            # 혼합 정밀도 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if i % 30 == 0:
                print(f"  혼합 정밀도 반복 {i}/{num_iterations}, Loss: {loss.item():.4f}")
    
    def compute_intensive_operations(self, num_iterations=800):
        """계산 집약적 연산 테스트"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return
        
        print(f"계산 집약적 연산 테스트 ({num_iterations} 반복)")
        
        for i in range(num_iterations):
            # 대용량 텐서 생성
            size = 2000
            a = torch.randn(size, size, device=self.device)
            b = torch.randn(size, size, device=self.device)
            
            # 복잡한 수학 연산들
            c = torch.matmul(a, b)
            d = torch.sin(c) * torch.cos(c) + torch.tan(torch.clamp(c, -1, 1))
            e = torch.exp(torch.clamp(d, -5, 5))
            f = torch.log(torch.abs(e) + 1e-8)
            g = torch.sqrt(torch.abs(f) + 1e-8)
            h = torch.pow(torch.abs(g), 0.5)
            
            # 통계적 연산
            mean_val = torch.mean(h)
            std_val = torch.std(h)
            max_val = torch.max(h)
            min_val = torch.min(h)
            
            # 소팅 연산
            if i % 20 == 0:
                sorted_vals, indices = torch.sort(h.flatten())
            
            # 조건부 연산
            mask = h > mean_val
            filtered = torch.where(mask, h, torch.zeros_like(h))
            
            if i % 80 == 0:
                print(f"  계산 집약적 반복 {i}/{num_iterations}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch GPU 사용률 집약적 테스트')
    parser.add_argument('--duration', type=int, default=240, help='테스트 지속 시간 (초)')
    parser.add_argument('--matrix-ops', type=int, default=500, help='행렬 연산 횟수')
    parser.add_argument('--conv-iterations', type=int, default=200, help='컨볼루션 반복 횟수')
    parser.add_argument('--transformer-iterations', type=int, default=100, help='트랜스포머 반복 횟수')
    parser.add_argument('--skip-conv', action='store_true', help='컨볼루션 테스트 건너뛰기')
    parser.add_argument('--skip-transformer', action='store_true', help='트랜스포머 테스트 건너뛰기')
    parser.add_argument('--skip-fft', action='store_true', help='FFT 테스트 건너뛰기')
    parser.add_argument('--skip-mixed-precision', action='store_true', help='혼합 정밀도 테스트 건너뛰기')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"\n🎮 GPU 정보:")
        print(f"  디바이스: {torch.cuda.get_device_name()}")
        print(f"  메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  멀티프로세서: {torch.cuda.get_device_properties(0).multi_processor_count}")
        print(f"  최대 스레드/블록: {torch.cuda.get_device_properties(0).max_threads_per_block}")
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print(f"\n🚀 PyTorch GPU 사용률 집약적 테스트 시작 (지속 시간: {args.duration}초)")
    print("=" * 60)
    
    # GPU 사용률 최대화 객체 생성
    gpu_maximizer = GPUUtilizationMaximizer()
    
    start_time = time.time()
    
    try:
        # 테스트 실행
        test_results = {}
        
        # 1. 병렬 행렬 연산
        print("\n1. 병렬 행렬 연산 테스트")
        gpu_maximizer.parallel_matrix_operations(
            num_operations=args.matrix_ops,
            matrix_size=1500
        )
        test_results['parallel_matrix_ops'] = True
        
        # 2. 컨볼루션 집약적 테스트
        if not args.skip_conv:
            print("\n2. 컨볼루션 집약적 테스트")
            gpu_maximizer.convolution_intensive_test(
                num_iterations=args.conv_iterations,
                batch_size=16
            )
            test_results['convolution_test'] = True
        
        # 3. 트랜스포머 집약적 테스트
        if not args.skip_transformer:
            print("\n3. 트랜스포머 집약적 테스트")
            gpu_maximizer.transformer_intensive_test(
                num_iterations=args.transformer_iterations,
                seq_len=256,
                batch_size=8
            )
            test_results['transformer_test'] = True
        
        # 4. FFT 집약적 테스트
        if not args.skip_fft:
            print("\n4. FFT 집약적 테스트")
            gpu_maximizer.fft_intensive_test(
                num_iterations=300,
                signal_size=4096
            )
            test_results['fft_test'] = True
        
        # 5. 혼합 정밀도 테스트
        if not args.skip_mixed_precision:
            print("\n5. 혼합 정밀도 테스트")
            gpu_maximizer.mixed_precision_test(num_iterations=150)
            test_results['mixed_precision_test'] = True
        
        # 6. 계산 집약적 연산
        print("\n6. 계산 집약적 연산 테스트")
        gpu_maximizer.compute_intensive_operations(num_iterations=300)
        test_results['compute_intensive_test'] = True
        
        # 남은 시간 동안 추가 병렬 연산 수행
        elapsed_time = time.time() - start_time
        remaining_time = args.duration - elapsed_time
        
        if remaining_time > 20:
            print(f"\n7. 추가 병렬 연산 ({remaining_time:.1f}초 동안)")
            extra_operations = max(100, int(remaining_time * 5))
            gpu_maximizer.parallel_matrix_operations(
                num_operations=extra_operations,
                matrix_size=1200
            )
            test_results['extra_operations'] = True
        
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
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_utilization = final_usage.get(f'gpu_{i}_utilization', 0)
                print(f"GPU {i} 사용률: {gpu_utilization}%")
        
        # GPU 메모리 정리
        print("\nGPU 메모리 정리 중...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("\n🎯 PyTorch GPU 사용률 집약적 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    main() 