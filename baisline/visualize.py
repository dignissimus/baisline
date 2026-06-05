import matplotlib.pyplot as plt
import numpy as np
from typing import List
from .reports import BenchmarkReport

# TODO: Rewrite/remove
def plot_roofline(
    reports: List[BenchmarkReport],
    hardware_max_bandwidth_gbps: float,
    hardware_max_tflops: float,
    save_path: str = "roofline.png"
):
    """
    Plots a Roofline chart comparing multiple model reports.
    
    Args:
        reports: List of BenchmarkReport objects.
        hardware_max_bandwidth_gbps: Max VRAM bandwidth in GB/s.
        hardware_max_tflops: Max compute performance in TFLOPs/s.
        save_path: Where to save the generated image.
    """
    plt.figure(figsize=(10, 7))
    
    ai_range = np.logspace(-2, 4, 100) 
    
    # Performance = min(AI * Bandwidth, max flops)
    bandwidth_tps = hardware_max_bandwidth_gbps / 1000.0
    roofline = np.minimum(ai_range * bandwidth_tps, hardware_max_tflops)
    
    plt.loglog(ai_range, roofline, 'k-', label="Roofline (Peak)")
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(reports)))
    has_valid_points = False
    for report, color in zip(reports, colors):
        ai = report.arithmetic_intensity
        perf = report.tflops_per_second
        
        plot_ai = ai if ai > 0 else 1e-3
        plot_perf = perf if perf > 0 else 1e-6
        
        plt.plot(plot_ai, plot_perf, marker='o', color=color, markersize=10, 
                 label=report.model_name, linestyle='None', markeredgecolor='black', zorder=5)
        
        if ai > 0 and perf > 0:
            has_valid_points = True
        else:
            print(f"Warning: {report.model_name} has zeros (AI={ai}, Perf={perf}). Plotting at chart edge.")

    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim(min(ai_range[0], 1e-2), max(ai_range[-1], 1e4))
    plt.ylim(min(roofline[0], 1e-5), max(hardware_max_tflops * 2, 1.0))

    plt.xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
    plt.ylabel('Performance (TFLOPs/second)', fontsize=12)
    plt.title('Roofline Model Comparison', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.text(ai_range[0], hardware_max_tflops * 1.1, f"Peak Compute: {hardware_max_tflops} TFLOPs/s", verticalalignment='bottom')
    plt.text(ai_range[10], ai_range[10] * bandwidth_tps * 1.5, f"Peak Bandwidth: {hardware_max_bandwidth_gbps} GB/s", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Roofline chart saved to {save_path}")
    plt.close()

# TODO: Refactor the below
def plot_flops_vs_metric(
    reports: List[BenchmarkReport],
    metric_name: str = "loss",
    save_path: str = "flops_vs_metric.png",
    use_log_scale: bool = True
):
    """
    Plots cumulative FLOPs vs a specific metric.
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(reports)))
    
    for report, color in zip(reports, colors):
        if not report.history:
            print(f"Warning: No history found for {report.model_name}. Skipping plot.")
            continue
            
        flops = [h["cum_flops"] for h in report.history]
        metrics = [h["metrics"].get(metric_name) for h in report.history]
        
        valid_points = [(f, m) for f, m in zip(flops, metrics) if m is not None]
        if not valid_points:
            print(f"Warning: Metric '{metric_name}' not found in history for {report.model_name}.")
            continue
            
        flops, metrics = zip(*valid_points)
        
        plt.plot(flops, metrics, marker='o', linestyle='-', color=color, label=report.model_name, markersize=4)

    if use_log_scale:
        plt.xscale('log')
    else:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.xlabel('Cumulative FLOPs', fontsize=12)
    plt.ylabel(metric_name.capitalize(), fontsize=12)
    plt.title(f'Training Progress: {metric_name.capitalize()} vs Cumulative FLOPs', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"FLOPs vs {metric_name} chart saved to {save_path}")
    plt.close()

def plot_steps_vs_metric(
    reports: List[BenchmarkReport],
    metric_name: str = "loss",
    save_path: str = "steps_vs_metric.png"
):
    """
    Plots training steps vs a specific metric.
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(reports)))
    
    for report, color in zip(reports, colors):
        if not report.history:
            print(f"Warning: No history found for {report.model_name}. Skipping plot.")
            continue
            
        steps = [h.get("step") for h in report.history]
        metrics = [h["metrics"].get(metric_name) for h in report.history]
        
        valid_points = [(s, m) for s, m in zip(steps, metrics) if s is not None and m is not None]
        if not valid_points:
            print(f"Warning: Metric '{metric_name}' or 'step' not found in history for {report.model_name}.")
            continue
            
        steps, metrics = zip(*valid_points)
        
        plt.plot(steps, metrics, marker='o', linestyle='-', color=color, label=report.model_name, markersize=4)

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(metric_name.capitalize(), fontsize=12)
    plt.title(f'Training Progress: {metric_name.capitalize()} vs Steps', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Steps vs {metric_name} chart saved to {save_path}")
    plt.close()
