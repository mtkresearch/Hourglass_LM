import json
import sys
import re

def transform_json(input_data):
    """
    轉換 JSON 格式
    - 只保留 "hourglass" 和 "conventional"
    - "conventional" 改名為 "conventional_w_Win"
    - 在 model_type 後加上參數數量後綴（從最外層 key 提取）
    - 提取 "psnr_batch" -> "overall" 的值,轉換為 "eval_psnr" 的 list
    - 將 "reps" 改為 "latent"，"mid" 改為 "hidden"
    """
    # 初始化輸出結構
    output_data = {
        "bs64_ep50_aug4": {}
    }
    
    for param_count_key, top_value in input_data.items():
        # 最外層的 key 就是參數數量
        param_count = param_count_key
        
        for model_type, model_data in top_value.items():
            # 只處理 hourglass 和 conventional
            if model_type == "hourglass":
                base_model_type = "hourglass"
            elif model_type == "normal_w_in_out":
                base_model_type = "conventional_w_Win"
            else:
                continue  # 跳過其他類型
            
            # 組合新的 model_type 名稱（加上參數數量後綴）
            new_model_type = f"{base_model_type}_{param_count}"
            
            if new_model_type not in output_data["bs64_ep50_aug4"]:
                output_data["bs64_ep50_aug4"][new_model_type] = {}
            
            # 遍歷配置層級
            for config_key, config_value in model_data.items():
                # 將 config_key 中的 "reps" 改為 "latent"，"mid" 改為 "hidden"
                new_config_key = config_key.replace("reps", "latent").replace("mid", "hidden")
                
                if new_config_key not in output_data["bs64_ep50_aug4"][new_model_type]:
                    output_data["bs64_ep50_aug4"][new_model_type][new_config_key] = {}
                
                for lr_key, lr_value in config_value.items():
                    if lr_key not in output_data["bs64_ep50_aug4"][new_model_type][new_config_key]:
                        output_data["bs64_ep50_aug4"][new_model_type][new_config_key][lr_key] = {}
                    
                    for run_key, run_value in lr_value.items():
                        # 只保留 metrics
                        if "metrics" in run_value and "psnr_batch" in run_value["metrics"]:
                            psnr_batch_value = run_value["metrics"]["psnr_batch"].get("overall")
                            
                            output_data["bs64_ep50_aug4"][new_model_type][new_config_key][lr_key][run_key] = {
                                "metrics": {
                                    # "eval_psnr": [psnr_batch_value] if psnr_batch_value is not None else [], 
                                    ## only for MNIST generative classification
                                    "test_psnr": run_value["metrics"]['psnr_batch']['overall']
                                }
                            }
    
    return output_data


def main():
    if len(sys.argv) != 3:
        print("使用方式: python script.py <input_json_path> <output_json_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        # 讀取輸入 JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # 轉換資料
        output_data = transform_json(input_data)
        
        # 寫入輸出 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"轉換成功!輸出檔案: {output_path}")
    
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 '{input_path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"錯誤: '{input_path}' 不是有效的 JSON 檔案")
        print(f"詳細錯誤: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()