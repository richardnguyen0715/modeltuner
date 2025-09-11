import torch
import os
import glob
import json
from datetime import datetime
from collections import defaultdict

def list_available_checkpoints():
    """List all available checkpoints with detailed information"""
    checkpoint_patterns = [
        'checkpoints/checkpoint_epoch_*.pth',
        'checkpoint_epoch_*.pth',
        'best_vqa_model.pth',
        'best_wups_model.pth',
        'best_fuzzy_model.pth',
        'final_model_*.pth',
        'interrupted_checkpoint_*.pth',
        'emergency_checkpoint_*.pth'
    ]
    
    checkpoints = []
    
    for pattern in checkpoint_patterns:
        for checkpoint_path in glob.glob(pattern):
            try:
                checkpoint_info = get_checkpoint_info(checkpoint_path)
                checkpoints.append(checkpoint_info)
            except Exception as e:
                print(f"Warning: Could not read checkpoint {checkpoint_path}: {e}")
    
    # Sort by epoch, then by modification time
    checkpoints.sort(key=lambda x: (x.get('epoch', -1), x.get('mtime', 0)))
    
    print(f"\n{'='*80}")
    print(f"AVAILABLE CHECKPOINTS")
    print(f"{'='*80}")
    
    if not checkpoints:
        print("No checkpoints found.")
        return []
    
    print(f"{'File':<40} {'Epoch':<6} {'Stage':<6} {'VQA':<6} {'WUPS':<6} {'Status':<12} {'Date'}")
    print(f"{'-'*80}")
    
    for checkpoint in checkpoints:
        filename = os.path.basename(checkpoint['path'])
        epoch = str(checkpoint.get('epoch', '?'))
        stage = str(checkpoint.get('stage', '?'))
        vqa = f"{checkpoint.get('vqa_score', 0):.3f}"
        wups = f"{checkpoint.get('wups_0.9', 0):.3f}"
        status = checkpoint.get('status', 'unknown')
        date = checkpoint.get('date', 'unknown')
        
        print(f"{filename:<40} {epoch:<6} {stage:<6} {vqa:<6} {wups:<6} {status:<12} {date}")
    
    return checkpoints

def get_checkpoint_info(checkpoint_path):
    """Get detailed information about a checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Basic info
        info = {
            'path': checkpoint_path,
            'epoch': checkpoint.get('epoch', None),
            'global_step': checkpoint.get('global_step', None),
            'stage': checkpoint.get('current_stage', None),
        }
        
        # File info
        stat = os.stat(checkpoint_path)
        info['mtime'] = stat.st_mtime
        info['size_mb'] = stat.st_size / (1024 * 1024)
        info['date'] = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
        
        # Best scores
        if 'best_scores' in checkpoint:
            best_scores = checkpoint['best_scores']
            info['vqa_score'] = best_scores.get('vqa_score', 0)
            info['wups_0.9'] = best_scores.get('wups_0.9', 0)
            info['fuzzy_accuracy'] = best_scores.get('fuzzy_accuracy', 0)
        
        # Training status
        resume_metadata = checkpoint.get('resume_metadata', {})
        training_complete = resume_metadata.get('training_complete', False)
        
        if training_complete:
            info['status'] = 'completed'
        elif 'interrupted' in checkpoint_path:
            info['status'] = 'interrupted'
        elif 'emergency' in checkpoint_path:
            info['status'] = 'emergency'
        elif 'best_' in checkpoint_path:
            info['status'] = 'best_model'
        else:
            info['status'] = 'checkpoint'
        
        return info
        
    except Exception as e:
        return {
            'path': checkpoint_path,
            'error': str(e),
            'status': 'error'
        }

def find_best_checkpoint(metric='vqa_score'):
    """Find the checkpoint with the best score for a given metric"""
    checkpoints = []
    
    for pattern in ['checkpoints/checkpoint_epoch_*.pth', 'best_*.pth']:
        for checkpoint_path in glob.glob(pattern):
            try:
                info = get_checkpoint_info(checkpoint_path)
                if metric in info:
                    checkpoints.append(info)
            except:
                continue
    
    if not checkpoints:
        return None
    
    best_checkpoint = max(checkpoints, key=lambda x: x.get(metric, 0))
    return best_checkpoint['path']

def cleanup_old_checkpoints(keep_last_n=5, keep_best_models=True):
    """Clean up old checkpoints, keeping only the most recent ones"""
    
    # Get all regular checkpoints
    checkpoint_files = glob.glob('checkpoints/checkpoint_epoch_*.pth')
    checkpoint_files.sort(key=os.path.getctime)  # Sort by creation time
    
    removed_count = 0
    
    # Remove old regular checkpoints
    if len(checkpoint_files) > keep_last_n:
        old_checkpoints = checkpoint_files[:-keep_last_n]
        
        for checkpoint_path in old_checkpoints:
            try:
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {os.path.basename(checkpoint_path)}")
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {checkpoint_path}: {e}")
    
    # Optionally keep best models
    if not keep_best_models:
        best_model_patterns = ['best_vqa_model.pth', 'best_wups_model.pth', 'best_fuzzy_model.pth']
        for pattern in best_model_patterns:
            if os.path.exists(pattern):
                try:
                    os.remove(pattern)
                    print(f"Removed best model: {pattern}")
                    removed_count += 1
                except Exception as e:
                    print(f"Failed to remove {pattern}: {e}")
    
    print(f"Cleanup completed. Removed {removed_count} files.")
    return removed_count

def resume_training_interactive():
    """Interactive helper for resuming training"""
    print(f"\n{'='*60}")
    print(f"INTERACTIVE RESUME TRAINING HELPER")
    print(f"{'='*60}")
    
    # List available checkpoints
    checkpoints = list_available_checkpoints()
    
    if not checkpoints:
        print("No checkpoints available for resuming.")
        return None
    
    print(f"\nOptions:")
    print(f"1. Auto-resume from latest checkpoint")
    print(f"2. Resume from specific checkpoint")
    print(f"3. Resume from best VQA score model")
    print(f"4. Resume from best WUPS model")
    print(f"5. Cancel")
    
    while True:
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                # Auto-resume from latest
                latest = max(checkpoints, key=lambda x: x.get('epoch', -1))
                return {
                    'resume_training': True,
                    'auto_resume': True,
                    'resume_from_checkpoint': 'latest'
                }
            
            elif choice == '2':
                # Specific checkpoint
                print(f"\nAvailable checkpoints:")
                for i, checkpoint in enumerate(checkpoints):
                    filename = os.path.basename(checkpoint['path'])
                    epoch = checkpoint.get('epoch', '?')
                    print(f"{i+1:2d}. {filename} (epoch {epoch})")
                
                idx = int(input("Select checkpoint number: ")) - 1
                if 0 <= idx < len(checkpoints):
                    return {
                        'resume_training': True,
                        'auto_resume': False,
                        'resume_from_checkpoint': checkpoints[idx]['path']
                    }
                else:
                    print("Invalid selection.")
                    continue
            
            elif choice == '3':
                # Best VQA
                best_vqa_path = find_best_checkpoint('vqa_score')
                if best_vqa_path:
                    return {
                        'resume_training': True,
                        'auto_resume': False,
                        'resume_from_checkpoint': best_vqa_path
                    }
                else:
                    print("No checkpoint with VQA score found.")
                    continue
            
            elif choice == '4':
                # Best WUPS
                best_wups_path = find_best_checkpoint('wups_0.9')
                if best_wups_path:
                    return {
                        'resume_training': True,
                        'auto_resume': False,
                        'resume_from_checkpoint': best_wups_path
                    }
                else:
                    print("No checkpoint with WUPS score found.")
                    continue
            
            elif choice == '5':
                return None
            
            else:
                print("Invalid choice. Please select 1-5.")
                continue
                
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled.")
            return None

if __name__ == "__main__":
    # Command line interface for checkpoint management
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python checkpoint_utils.py list")
        print("  python checkpoint_utils.py cleanup [keep_n]")
        print("  python checkpoint_utils.py best [metric]")
        print("  python checkpoint_utils.py interactive")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'list':
        list_available_checkpoints()
    
    elif command == 'cleanup':
        keep_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        cleanup_old_checkpoints(keep_n)
    
    elif command == 'best':
        metric = sys.argv[2] if len(sys.argv) > 2 else 'vqa_score'
        best_path = find_best_checkpoint(metric)
        print(f"Best checkpoint for {metric}: {best_path}")
    
    elif command == 'interactive':
        config = resume_training_interactive()
        if config:
            print(f"\nRecommended configuration:")
            for key, value in config.items():
                print(f"  config['{key}'] = {repr(value)}")
        else:
            print("Resume cancelled.")
    
    else:
        print(f"Unknown command: {command}")