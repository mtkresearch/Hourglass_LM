import json
import sys
from pathlib import Path

def merge_json_files(json_files):
    """
    合併多個 JSON 檔案
    假設每個 JSON 都有相同的結構,如 {"bs128_ep30": {...}}
    會將所有檔案的內容深度合併
    """
    merged_data = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 深度合併
            for top_key, top_value in data.items():
                if top_key not in merged_data:
                    merged_data[top_key] = {}
                
                for model_type, model_data in top_value.items():
                    if model_type not in merged_data[top_key]:
                        merged_data[top_key][model_type] = {}
                    
                    for config_key, config_value in model_data.items():
                        if config_key not in merged_data[top_key][model_type]:
                            merged_data[top_key][model_type][config_key] = {}
                        
                        for lr_key, lr_value in config_value.items():
                            if lr_key not in merged_data[top_key][model_type][config_key]:
                                merged_data[top_key][model_type][config_key][lr_key] = {}
                            
                            for run_key, run_value in lr_value.items():
                                # 直接覆寫或新增 run 資料
                                merged_data[top_key][model_type][config_key][lr_key][run_key] = run_value
            
            print(f"已合併: {json_file}")
        
        except FileNotFoundError:
            print(f"警告: 找不到檔案 '{json_file}',跳過")
        except json.JSONDecodeError:
            print(f"警告: '{json_file}' 不是有效的 JSON 檔案,跳過")
        except Exception as e:
            print(f"警告: 處理 '{json_file}' 時發生錯誤: {str(e)},跳過")
    
    return merged_data


def main():
    if len(sys.argv) < 3:
        print("使用方式: python merge_json.py <output_json_path> <input_json_1> <input_json_2> ...")
        print("範例: python merge_json.py merged.json file1.json file2.json file3.json")
        sys.exit(1)
    
    output_path = sys.argv[1]
    input_files = sys.argv[2:]
    
    if len(input_files) == 0:
        print("錯誤: 請至少提供一個輸入 JSON 檔案")
        sys.exit(1)
    
    print(f"準備合併 {len(input_files)} 個檔案...")
    
    try:
        # 合併所有檔案
        merged_data = merge_json_files(input_files)
        
        # 寫入輸出 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n合併成功!輸出檔案: {output_path}")
    
    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()