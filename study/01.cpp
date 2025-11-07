#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        // 创建哈希表存储<数值, 对应下标>
        unordered_map<int, int> num_map;
        
        // 遍历数组
        for (int i = 0; i < nums.size(); ++i) {
            // 计算当前需要的补数
            int complement = target - nums[i];
            
            // 检查补数是否存在于哈希表中
            if (num_map.find(complement) != num_map.end()) {
                // 找到则返回两个下标（补数的下标在前）
                return {num_map[complement], i};
            }
            
            // 未找到则将当前数值与下标存入哈希表
            num_map[nums[i]] = i;
        }
        
        // 根据题目假设，此处不会执行（保证有解）
        return {};
    }
};

#include <stdio.h>
int main()
{
    int a = sizeof(int);
    printf("%d\n",a);

    return 0;
}