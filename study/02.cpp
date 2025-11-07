#include <vector>
#include <string>
#include <unordered_map>
#include <array>
#include <iostream> // 提交时可以包含必要的头文件

using namespace std;

class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) 
	{
        unordered_map<string, vector<string>> mp;

        for (string& s : strs) {
            array<int, 26> counts = {0};
            for (char c : s) {
                counts[c - 'a']++;
            }

            string key = "";
            for (int count : counts) {
                key += to_string(count) + "#";
            }
            mp[key].push_back(s);
        }

        vector<vector<string>> result;
        for (auto& it : mp) {
            result.push_back(it.second);
        }
        return result;
    }
};