"""
Kod Kalite Analizi - Kiviat (Radar) Grafiği ve Block Histogram
Source Monitor alternatifi olarak Python ile analiz
"""

import os
import ast
import math
import matplotlib.pyplot as plt
import numpy as np
from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit
from radon.raw import analyze

# Analiz edilecek dosyalar
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_FILES = [
    'rl_module_v2.py',
    'analiz_module.py', 
    'boundary_analysis.py',
    'llm_module.py',
    'main.py',
    'train_v2.py'
]

def analyze_file(filepath):
    """Tek bir dosyayı analiz et"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Raw metrics (LOC, SLOC, comments, etc.)
        raw = analyze(code)
        
        # Cyclomatic Complexity
        cc_results = cc_visit(code)
        avg_complexity = np.mean([block.complexity for block in cc_results]) if cc_results else 0
        max_complexity = max([block.complexity for block in cc_results]) if cc_results else 0
        
        # Maintainability Index
        mi_score = mi_visit(code, True)
        
        # Halstead metrics
        halstead = h_visit(code)
        
        # AST analysis
        tree = ast.parse(code)
        num_functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        num_classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        num_imports = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        
        return {
            'filename': os.path.basename(filepath),
            'loc': raw.loc,  # Total lines
            'sloc': raw.sloc,  # Source lines (no comments/blanks)
            'comments': raw.comments,
            'blank': raw.blank,
            'avg_complexity': round(avg_complexity, 2),
            'max_complexity': max_complexity,
            'maintainability_index': round(mi_score, 2),
            'num_functions': num_functions,
            'num_classes': num_classes,
            'num_imports': num_imports,
            'comment_ratio': round((raw.comments / raw.loc * 100) if raw.loc > 0 else 0, 2)
        }
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return None

def create_kiviat_chart(metrics_list, output_path='kiviat_chart.png'):
    """Kiviat (Radar) grafiği oluştur"""
    
    # Normalize edilecek metrikler
    categories = ['LOC\n(normalized)', 'Complexity', 'Maintainability', 'Functions', 'Comment\nRatio']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Her dosya için radar çiz
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Kapatmak için
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_list)))
    
    for idx, metrics in enumerate(metrics_list):
        if metrics is None:
            continue
            
        # Normalize değerler (0-100 arası)
        values = [
            min(metrics['sloc'] / 20, 100),  # LOC normalized
            min(metrics['avg_complexity'] * 10, 100),  # Complexity
            metrics['maintainability_index'],  # MI already 0-100
            min(metrics['num_functions'] * 2, 100),  # Functions
            min(metrics['comment_ratio'] * 5, 100)  # Comment ratio
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=metrics['filename'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 100)
    ax.set_title('Kod Kalite Metrikleri - Kiviat Grafiği\n(Source Monitor Alternatifi)', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Kiviat grafiği kaydedildi: {output_path}")

def create_block_histogram(metrics_list, output_path='block_histogram.png'):
    """Block Histogram oluştur - Complexity dağılımı"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. LOC Histogram
    ax1 = axes[0, 0]
    filenames = [m['filename'][:15] for m in metrics_list if m]
    locs = [m['sloc'] for m in metrics_list if m]
    colors = ['green' if loc < 200 else 'orange' if loc < 500 else 'red' for loc in locs]
    ax1.bar(filenames, locs, color=colors, edgecolor='black')
    ax1.set_title('Source Lines of Code (SLOC)', fontweight='bold')
    ax1.set_ylabel('Lines')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Good (<200)')
    ax1.axhline(y=500, color='red', linestyle='--', alpha=0.7, label='Warning (>500)')
    ax1.legend()
    
    # 2. Complexity Histogram
    ax2 = axes[0, 1]
    complexities = [m['avg_complexity'] for m in metrics_list if m]
    colors = ['green' if c < 5 else 'orange' if c < 10 else 'red' for c in complexities]
    ax2.bar(filenames, complexities, color=colors, edgecolor='black')
    ax2.set_title('Average Cyclomatic Complexity', fontweight='bold')
    ax2.set_ylabel('Complexity')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Good (<5)')
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Warning (>10)')
    ax2.legend()
    
    # 3. Maintainability Index
    ax3 = axes[1, 0]
    mi_scores = [m['maintainability_index'] for m in metrics_list if m]
    colors = ['green' if mi > 65 else 'orange' if mi > 40 else 'red' for mi in mi_scores]
    ax3.bar(filenames, mi_scores, color=colors, edgecolor='black')
    ax3.set_title('Maintainability Index', fontweight='bold')
    ax3.set_ylabel('MI Score (0-100)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=65, color='green', linestyle='--', alpha=0.7, label='Good (>65)')
    ax3.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Warning (<40)')
    ax3.legend()
    
    # 4. Functions per File
    ax4 = axes[1, 1]
    functions = [m['num_functions'] for m in metrics_list if m]
    colors = ['green' if f < 15 else 'orange' if f < 30 else 'red' for f in functions]
    ax4.bar(filenames, functions, color=colors, edgecolor='black')
    ax4.set_title('Number of Functions per File', fontweight='bold')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Block Histogram - Kod Kalite Analizi\n(Source Monitor Alternatifi)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Block histogram kaydedildi: {output_path}")

def create_summary_table(metrics_list, output_path='metrics_summary.png'):
    """Özet tablo oluştur"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Tablo verileri
    headers = ['Dosya', 'LOC', 'SLOC', 'Complexity', 'MI Score', 'Functions', 'Classes', 'Comments%']
    data = []
    
    for m in metrics_list:
        if m:
            data.append([
                m['filename'],
                m['loc'],
                m['sloc'],
                m['avg_complexity'],
                m['maintainability_index'],
                m['num_functions'],
                m['num_classes'],
                f"{m['comment_ratio']}%"
            ])
    
    # Toplam satırı
    if metrics_list:
        valid_metrics = [m for m in metrics_list if m]
        data.append([
            'TOPLAM',
            sum(m['loc'] for m in valid_metrics),
            sum(m['sloc'] for m in valid_metrics),
            round(np.mean([m['avg_complexity'] for m in valid_metrics]), 2),
            round(np.mean([m['maintainability_index'] for m in valid_metrics]), 2),
            sum(m['num_functions'] for m in valid_metrics),
            sum(m['num_classes'] for m in valid_metrics),
            '-'
        ])
    
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(headers)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header renkleri
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Son satır (toplam) vurgula
    for i in range(len(headers)):
        table[(len(data), i)].set_facecolor('#E2EFDA')
    
    plt.title('Kod Metrikleri Özet Tablosu', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Özet tablo kaydedildi: {output_path}")

def main():
    print("=" * 60)
    print("KOD KALİTE ANALİZİ - SOURCE MONITOR ALTERNATİFİ")
    print("=" * 60)
    
    # Dosyaları analiz et
    metrics_list = []
    for filename in PYTHON_FILES:
        filepath = os.path.join(PROJECT_DIR, filename)
        if os.path.exists(filepath):
            print(f"📊 Analiz ediliyor: {filename}")
            metrics = analyze_file(filepath)
            if metrics:
                metrics_list.append(metrics)
                print(f"   LOC: {metrics['loc']}, Complexity: {metrics['avg_complexity']}, MI: {metrics['maintainability_index']}")
    
    if not metrics_list:
        print("❌ Analiz edilecek dosya bulunamadı!")
        return
    
    print("\n" + "=" * 60)
    print("GRAFİKLER OLUŞTURULUYOR...")
    print("=" * 60)
    
    # Grafikleri oluştur
    create_kiviat_chart(metrics_list, os.path.join(PROJECT_DIR, 'kiviat_chart.png'))
    create_block_histogram(metrics_list, os.path.join(PROJECT_DIR, 'block_histogram.png'))
    create_summary_table(metrics_list, os.path.join(PROJECT_DIR, 'metrics_summary.png'))
    
    print("\n" + "=" * 60)
    print("✅ ANALİZ TAMAMLANDI!")
    print("=" * 60)
    print("\nOluşturulan dosyalar:")
    print("  📈 kiviat_chart.png - Kiviat (Radar) Grafiği")
    print("  📊 block_histogram.png - Block Histogram")
    print("  📋 metrics_summary.png - Özet Tablo")
    print("\nBu dosyaları raporuna ekleyebilirsin!")

if __name__ == "__main__":
    main()
