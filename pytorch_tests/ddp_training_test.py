"""
PyTorch DDP 분산 학습 테스트
이 테스트는 가짜 데이터셋을 사용하여 분산 학습을 수행하고 리소스 모니터링을 테스트합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import numpy as np
import time
import argparse
import sys
import os
from pathlib import Path

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


class LazyFakeDataset(Dataset):
    """
    가짜 데이터셋 - 실제 이미지 대신 무작위 텐서와 라벨을 생성하여 메모리 사용 최소화
    """
    def __init__(self, num_samples=100000, image_size=(3, 224, 224), num_classes=100):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 무작위 이미지 생성 (3x224x224)
        image = torch.randn(self.image_size, dtype=torch.float32)
        # 0~99 사이의 무작위 라벨 생성
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


def setup(rank, world_size, use_cpu=False):
    """DDP 환경 초기화"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    backend = "gloo" if use_cpu else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    if not use_cpu:
        torch.cuda.set_device(rank)


def cleanup():
    """DDP 환경 정리"""
    dist.destroy_process_group()


def create_model(num_classes=100):
    """복잡한 ResNet 모델 생성"""
    model = models.resnet50(pretrained=False)
    # 출력 클래스 수 조정
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(rank, world_size, args):
    """분산 학습 함수"""
    print(f"Running DDP training on rank {rank}")
    
    # CPU/GPU 설정
    use_cpu = not torch.cuda.is_available() or args.force_cpu
    device = torch.device('cpu' if use_cpu else f'cuda:{rank}')
    
    # DDP 환경 설정
    setup(rank, world_size, use_cpu)
    
    # 모델 생성 및 디바이스로 이동
    model = create_model(num_classes=args.num_classes)
    model = model.to(device)
    
    if use_cpu:
        model = DDP(model)
    else:
        model = DDP(model, device_ids=[rank])
    
    # 데이터셋 및 데이터로더 설정
    dataset = LazyFakeDataset(
        num_samples=args.num_samples,
        image_size=(3, 224, 224),
        num_classes=args.num_classes
    )
    
    # 분산 샘플러 사용
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2 if use_cpu else 4,
        pin_memory=not use_cpu
    )
    
    # 옵티마이저 및 손실함수 설정
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 리소스 모니터링 시작 (rank 0에서만)
    monitor = None
    if rank == 0:
        monitor = ResourceMonitor(interval=0.5)
        monitor.start_monitoring()
        device_info = "CPU" if use_cpu else f"GPU {world_size}개"
        print(f"\n🔥 PyTorch DDP 학습 시작 ({device_info} 사용)")
        print(f"   샘플 수: {args.num_samples:,}")
        print(f"   에폭 수: {args.epochs}")
        print(f"   배치 크기: {args.batch_size}")
        print(f"   클래스 수: {args.num_classes}")
        print(f"   디바이스: {device}")
        print("=" * 60)
    
    # 학습 루프
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 에폭마다 샘플러 설정
        
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device, non_blocking=not use_cpu)
            target = target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 진행상황 출력 (rank 0에서만)
            if rank == 0 and batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # 에폭 완료 후 정보 출력
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches
        
        if rank == 0:
            print(f"✅ Epoch {epoch+1}/{args.epochs} 완료 - "
                  f"평균 Loss: {avg_loss:.4f}, 시간: {epoch_time:.2f}초")
        
        # 학습률 업데이트
        scheduler.step()
        
        # 중간 체크포인트 저장 (rank 0에서만, 10 에폭마다)
        if rank == 0 and (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
    
    # 최종 모델 저장 (rank 0에서만)
    if rank == 0:
        save_model(model, args.save_dir, "model_final.pth")
        print(f"\n🎯 모델 저장 완료: {args.save_dir}/model_final.pth")
        
        # 리소스 모니터링 종료
        if monitor:
            monitor.stop_monitoring()
            monitor.print_summary()
            
            # 최종 리소스 사용량 출력
            final_usage = monitor.get_current_usage()
            print(f"\n최종 리소스 사용량:")
            print(f"CPU: {final_usage.get('cpu_cores', 0):.2f} cores")
            print(f"Memory: {final_usage.get('memory_gb', 0):.2f} GB")
            
            # GPU 사용량 출력 (GPU 사용 시만)
            if not use_cpu:
                for i in range(world_size):
                    gpu_util = final_usage.get(f'gpu_{i}_utilization', 0)
                    gpu_mem = final_usage.get(f'gpu_{i}_memory_used_gb', 0)
                    if gpu_util > 0 or gpu_mem > 0:
                        print(f"GPU {i}: {gpu_util:.1f}% 사용률, {gpu_mem:.2f} GB 메모리")
    
    cleanup()


def save_model(model, save_dir, filename):
    """모델 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # DDP 모델의 경우 module을 사용하여 원본 모델 저장
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    torch.save(model_state, save_dir / filename)
    print(f"모델 저장: {save_dir / filename}")


def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    """체크포인트 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # DDP 모델의 경우 module을 사용하여 원본 모델 저장
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, save_dir / filename)
    print(f"체크포인트 저장: {save_dir / filename}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch DDP 분산 학습 테스트')
    parser.add_argument('--num-samples', type=int, default=100000, help='데이터셋 샘플 수')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.01, help='학습률')
    parser.add_argument('--num-classes', type=int, default=100, help='클래스 수')
    parser.add_argument('--save-dir', type=str, default='./saved_models', help='모델 저장 디렉토리')
    parser.add_argument('--force-single-gpu', action='store_true', help='단일 GPU 강제 사용')
    parser.add_argument('--force-cpu', action='store_true', help='CPU 강제 사용')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # CPU/GPU 설정
    if args.force_cpu:
        print("⚠️  CPU 모드로 강제 실행합니다.")
        world_size = 1
    elif not torch.cuda.is_available():
        print("⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        world_size = 1
        args.force_cpu = True
    else:
        world_size = torch.cuda.device_count()
        print(f"🔍 사용 가능한 GPU 개수: {world_size}")
        
        # 분산 학습 실행 조건 확인
        if world_size < 2 and not args.force_single_gpu:
            print("❌ 분산 학습을 위해서는 2개 이상의 GPU가 필요합니다.")
            print("   단일 GPU에서 테스트하려면 --force-single-gpu 옵션을 사용하세요.")
            print("   CPU에서 테스트하려면 --force-cpu 옵션을 사용하세요.")
            return
        elif args.force_single_gpu:
            world_size = 1
            print("⚠️  단일 GPU 모드로 실행합니다.")
    
    device_info = "CPU" if args.force_cpu else f"{world_size}개 GPU"
    print(f"\n🚀 {device_info}로 DDP 학습 시작")
    print("=" * 60)
    
    try:
        if world_size == 1:
            # 단일 디바이스 학습
            train(0, 1, args)
        else:
            # 멀티 GPU 분산 학습
            mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
            
        print("\n✅ DDP 학습이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)


if __name__ == "__main__":
    main() 