import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-grade medical ML plots
sns.set_theme(style='whitegrid', font='sans-serif')
palette = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d']

os.makedirs('reports/plots', exist_ok=True)
df = pd.read_csv('dataset_audit.csv')

print('--- AUDIT METRICS DEEP DIVE ---')
total_images = len(df)
print(f'Total images: {total_images}')

# Patients
unique_patients_overall = df['patient_id_if_available'].nunique()
print(f'Unique patients: {unique_patients_overall}')

train_patients = set(df[df['split'] == 'train']['patient_id_if_available'])
valid_patients = set(df[df['split'] == 'valid']['patient_id_if_available'])
test_patients = set(df[df['split'] == 'test']['patient_id_if_available'])

print(f'Train patients: {len(train_patients)}')
print(f'Valid patients: {len(valid_patients)}')
print(f'Test patients: {len(test_patients)}')

train_val_overlap = train_patients.intersection(valid_patients)
train_test_overlap = train_patients.intersection(test_patients)
val_test_overlap = valid_patients.intersection(test_patients)

print(f'Train-Valid patient overlap: {len(train_val_overlap)}')
print(f'Train-Test patient overlap: {len(train_test_overlap)}')
print(f'Valid-Test patient overlap: {len(val_test_overlap)}')

# Formats & Channels & Bit depth
formats = df['file_format'].value_counts().to_dict()
channels = df['channels'].value_counts().to_dict()
bit_depths = df['bit_depth'].value_counts().to_dict()
modes = df['mode'].value_counts().to_dict()
print(f'Formats: {formats}')
print(f'Channels: {channels}')
print(f'Bit Depths: {bit_depths}')
print(f'Modes: {modes}')

# Dimensions
dim_stats = {
    'min_width': int(df['width'].min()),
    'max_width': int(df['width'].max()),
    'mean_width': float(df['width'].mean()),
    'median_width': float(df['width'].median()),
    'min_height': int(df['height'].min()),
    'max_height': int(df['height'].max()),
    'mean_height': float(df['height'].mean()),
    'median_height': float(df['height'].median()),
    'min_ar': float(df['aspect_ratio'].min()),
    'max_ar': float(df['aspect_ratio'].max()),
    'mean_ar': float(df['aspect_ratio'].mean()),
    'median_ar': float(df['aspect_ratio'].median()),
}
print(f'Dimension stats: {dim_stats}')

# Regions & Labels
regions = df['anatomical_region'].value_counts().to_dict()
labels = df['fracture_label'].value_counts().to_dict()
print(f'Regions: {regions}')
print(f'Fracture labels: {labels}')

# Region x Fracture breakdown
cross_tab = pd.crosstab(df['anatomical_region'], df['fracture_label'], margins=True)
print('\nRegion x Fracture Crosstab:')
print(cross_tab)

# Duplicates
total_duplicates = df['is_duplicate'].sum()
unique_duplicate_hashes = df[df['is_duplicate']]['md5_hash'].nunique()
cross_split_dups = df['cross_split_duplicate'].sum()
cross_patient_dups = df['cross_patient_duplicate'].sum()
cross_label_conflicts = df['cross_label_conflict'].sum()
print(f'\nTotal duplicate rows: {total_duplicates}')
print(f'Unique image hashes with duplicates: {unique_duplicate_hashes}')
print(f'Cross-split duplicate rows: {cross_split_dups}')
print(f'Cross-patient duplicate rows: {cross_patient_dups}')
print(f'Cross-label conflicting rows: {cross_label_conflicts}')

# Corrupted images
corrupt_count = df['is_corrupt'].sum()
print(f'Corrupt images: {corrupt_count}')

# Missing labels
missing_labels = df['fracture_label'].isna().sum() + (df['fracture_label'] == 'None').sum()
missing_regions = df['anatomical_region'].isna().sum() + (df['anatomical_region'] == 'Unknown').sum()
print(f'Missing labels: {missing_labels}, Missing regions: {missing_regions}')

# Suspicious records
suspicious_count = df['is_suspicious'].sum()
print(f'Suspicious records: {suspicious_count}')
quest = pd.read_csv('reports/questionable_records.csv')
print(f'Total questionable records in report: {len(quest)}')
print('Questionable breakdown:')
print(quest['suspicion_reasons'].value_counts().head(10))

# --- PLOT 1: Images per Anatomical Region ---
plt.figure(figsize=(10, 6), dpi=300)
region_order = df['anatomical_region'].value_counts().index
ax = sns.countplot(data=df, x='anatomical_region', order=region_order, palette='crest')
plt.title('BoneFract Dataset: Total Images per Anatomical Region', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Anatomical Region', fontsize=12, labelpad=10)
plt.ylabel('Image Count', fontsize=12, labelpad=10)
plt.xticks(rotation=15, ha='right', fontsize=11)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f'{int(h):,}', (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points', fontweight='bold')
plt.tight_layout()
p1_path = 'reports/plots/01_images_per_anatomical_region.png'
plt.savefig(p1_path)
plt.close()
print(f'Saved {p1_path}')

# --- PLOT 2: Fracture vs Non-Fracture Distribution ---
plt.figure(figsize=(8, 6), dpi=300)
colors = ['#2ca02c', '#d62728'] # Green for Negative, Red for Positive
ax = sns.countplot(data=df, x='fracture_label', palette=colors, order=['Negative', 'Positive'])
plt.title('BoneFract Dataset: Overall Fracture Class Distribution', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Fracture Status', fontsize=12, labelpad=10)
plt.ylabel('Image Count', fontsize=12, labelpad=10)
for p in ax.patches:
    h = p.get_height()
    pct = (h / total_images) * 100
    ax.annotate(f'{int(h):,} ({pct:.1f}%)', (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom', fontsize=11, xytext=(0, 5), textcoords='offset points', fontweight='bold')
plt.tight_layout()
p2_path = 'reports/plots/02_fracture_vs_non_fracture_overall.png'
plt.savefig(p2_path)
plt.close()
print(f'Saved {p2_path}')

# --- PLOT 3: Fracture vs Non-Fracture within Every Anatomical Region ---
plt.figure(figsize=(12, 6), dpi=300)
ax = sns.countplot(data=df, x='anatomical_region', hue='fracture_label', order=region_order, palette=['#2ca02c', '#d62728'])
plt.title('BoneFract Dataset: Fracture vs Non-Fracture Distribution by Anatomical Region', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Anatomical Region', fontsize=12, labelpad=10)
plt.ylabel('Image Count', fontsize=12, labelpad=10)
plt.legend(title='Condition', labels=['Negative (Normal)', 'Positive (Fractured)'], frameon=True)
plt.xticks(rotation=15, ha='right', fontsize=11)
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f'{int(h):,}', (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
p3_path = 'reports/plots/03_fracture_by_anatomical_region.png'
plt.savefig(p3_path)
plt.close()
print(f'Saved {p3_path}')

# --- PLOT 4: Image Dimension Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
# Subplot 4a: Scatter/KDE of Width vs Height
sns.histplot(data=df, x='width', y='height', bins=40, cbar=True, cmap='viridis', ax=axes[0])
axes[0].set_title('Width vs Height Distribution (Heatmap)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Width (pixels)', fontsize=11)
axes[0].set_ylabel('Height (pixels)', fontsize=11)
axes[0].axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.7, label='1:1 Aspect Ratio')
axes[0].legend(loc='upper left')

# Subplot 4b: Aspect Ratio Distribution
sns.histplot(df['aspect_ratio'], bins=50, kde=True, color='#1f77b4', ax=axes[1])
axes[1].axvline(1.0, color='red', linestyle='--', label='Square (1.0)')
axes[1].set_title('Aspect Ratio Distribution (Width / Height)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Aspect Ratio', fontsize=11)
axes[1].set_ylabel('Image Count', fontsize=11)
axes[1].legend(loc='upper right')

plt.suptitle('BoneFract Dataset: Image Geometry Profiles', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
p4_path = 'reports/plots/04_image_dimension_distribution.png'
plt.savefig(p4_path)
plt.close()
print(f'Saved {p4_path}')

# Output detailed summary metrics to text for reporting
with open('reports/audit_summary_metrics.txt', 'w') as f:
    f.write(f'TOTAL_IMAGES: {total_images}\n')
    f.write(f'UNIQUE_PATIENTS: {unique_patients_overall}\n')
    f.write(f'TRAIN_PATIENTS: {len(train_patients)}\n')
    f.write(f'VALID_PATIENTS: {len(valid_patients)}\n')
    f.write(f'TEST_PATIENTS: {len(test_patients)}\n')
    f.write(f'TRAIN_VAL_OVERLAP: {len(train_val_overlap)}\n')
    f.write(f'TRAIN_TEST_OVERLAP: {len(train_test_overlap)}\n')
    f.write(f'VAL_TEST_OVERLAP: {len(val_test_overlap)}\n')
    f.write(f'TOTAL_DUPLICATES: {total_duplicates}\n')
    f.write(f'UNIQUE_DUPLICATE_HASHES: {unique_duplicate_hashes}\n')
    f.write(f'CROSS_SPLIT_DUPLICATES: {cross_split_dups}\n')
    f.write(f'CROSS_PATIENT_DUPLICATES: {cross_patient_dups}\n')
    f.write(f'CROSS_LABEL_CONFLICTS: {cross_label_conflicts}\n')
    f.write(f'CORRUPT_COUNT: {corrupt_count}\n')
    f.write(f'QUESTIONABLE_RECORDS: {len(quest)}\n')
    f.write(f'DIM_STATS: {dim_stats}\n')
    f.write(f'FORMATS: {formats}\n')
    f.write(f'CHANNELS: {channels}\n')
    f.write(f'BIT_DEPTHS: {bit_depths}\n')
    f.write(f'MODES: {modes}\n')
    f.write(f'REGIONS: {regions}\n')
    f.write(f'LABELS: {labels}\n')
print('Metrics saved to reports/audit_summary_metrics.txt')
