import os
import pandas as pd

def main():
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('validation.csv')
    test_df = pd.read_csv('test.csv')

    all_df = pd.concat([
        train_df.assign(split='train'),
        val_df.assign(split='val'),
        test_df.assign(split='test')
    ])
    
    unique_regions = sorted(train_df['anatomical_region'].unique())

    report_rows = []
    print(f"{'class_id':<10} {'class_name':<15} {'training_count':<15} {'validation_count':<18} {'test_count':<12}")
    print("-" * 72)

    for idx, reg in enumerate(unique_regions):
        t_cnt = int((train_df['anatomical_region'] == reg).sum())
        v_cnt = int((val_df['anatomical_region'] == reg).sum())
        te_cnt = int((test_df['anatomical_region'] == reg).sum())
        
        pos_cnt = int(((all_df['anatomical_region'] == reg) & (all_df['fracture_label'] == 'Positive')).sum())
        neg_cnt = int(((all_df['anatomical_region'] == reg) & (all_df['fracture_label'] == 'Negative')).sum())
        
        print(f"{idx:<10} {reg:<15} {t_cnt:<15} {v_cnt:<18} {te_cnt:<12}")
        
        report_rows.append({
            'class_id': idx,
            'class_name': reg,
            'number_of_training_images': t_cnt,
            'number_of_validation_images': v_cnt,
            'number_of_test_images': te_cnt,
            'fracture_positive_count': pos_cnt,
            'fracture_negative_count': neg_cnt
        })

    rep_df = pd.DataFrame(report_rows)
    os.makedirs('reports', exist_ok=True)
    rep_df.to_csv('reports/class_definition_report.csv', index=False)
    print("\nSaved reports/class_definition_report.csv successfully.")

if __name__ == '__main__':
    main()
