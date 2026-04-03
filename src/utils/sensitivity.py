import torch
import torch.nn.functional as F
import numpy as np

def calculate_stability(model, input_tensor, num_samples=10, noise_std=0.01):
    """
    Calculates the stability (inverse of variance) of a model's prediction 
    under small input perturbations. A proxy for uncertainty.
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Baseline prediction
        base_probs = F.softmax(model(input_tensor), dim=1).cpu().numpy()
        
        # Perturbed predictions
        predictions = []
        for _ in range(num_samples):
            noise = torch.randn_like(input_tensor) * noise_std
            perturbed_input = input_tensor + noise
            probs = F.softmax(model(perturbed_input), dim=1).cpu().numpy()
            predictions.append(probs)
            
        predictions = np.array(predictions)
        
        # Calculate variance across samples
        variance = np.var(predictions, axis=0).mean() # Mean variance across classes
        
        # Stability score: 1 - variance (clamped)
        # Variance is typically very small, so we might need a scaling factor
        stability_score = max(0, 1 - (variance * 100)) # Simple heuristic
        
        return {
            'mean_prob': np.mean(predictions, axis=0),
            'std_prob': np.std(predictions, axis=0),
            'stability_score': stability_score,
            'variance': variance
        }
