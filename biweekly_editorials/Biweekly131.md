## Question - 1 :  Find the XOR of Numbers Which Appear Twice

<details>
<summary>Python</summary>

```python
class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        d = defaultdict(int)
        ans = 0
        for i in range(len(nums)):
            d[nums[i]]+=1
            if d[nums[i]] == 2:
                ans^=nums[i]
        return ans
```

</details>

<details>
<summary>Cpp</summary>

```cpp
class Solution {
public:
    int duplicateNumbersXOR(vector<int>& nums) {
        int d = 0;
        map<int, int> m;
        for (int i : nums) {
            m[i]++;
        }
        for (auto i : m) {
            if (i.second > 1)
                d = d ^ i.first;
        }

        return d;
    }
};
```

</details>

<details>
<summary>Java</summary>

```java
class Solution {
    public int duplicateNumbersXOR(int[] nums) {
        int d = 0;
        Map<Integer, Integer> m = new HashMap<>();
        for (int i : nums) {
            m.put(i, m.getOrDefault(i, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : m.entrySet()) {
            if (entry.getValue() > 1) {
                d = d ^ entry.getKey();
            }
        }

        return d;
    }
```

</details>

## Question - 2 : Find Occurrences of an Element in an Array

<details>
<summary>Python</summary>

```python
class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        d = dict()
        cnt = 1
        for i in range(len(nums)):
            if nums[i] == x:
                d[cnt] = i
                cnt+=1
        ans = [0] * len(queries)
        for i in range(len(queries)):
            ans[i] = d.get(queries[i],-1)
        return ans
```

</details>

<details>
<summary>Cpp</summary>

```cpp
class Solution {
public:
    vector<int> occurrencesOfElement(vector<int>& nums, vector<int>& queries, int x) {
        map<int, int> m;

        int count = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == x) {
                count++;
                m[count] = i;
            }
        }

        vector<int> v;
        for (int i = 0; i < queries.size(); i++) {
            if (queries[i] > count) {
                v.push_back(-1);
            }
            else {
                v.push_back(m[queries[i]]);
            }
        }
        return v;
    }
};
```

</details>

<details>
<summary>Java</summary>

```java
class Solution {
    public List<Integer> occurrencesOfElement(List<Integer> nums, List<Integer> queries, int x) {
        Map<Integer, Integer> m = new HashMap<>();

        int count = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums.get(i) == x) {
                count++;
                m.put(count, i);
            }
        }

        List<Integer> v = new ArrayList<>();
        for (int i = 0; i < queries.size(); i++) {
            if (queries.get(i) > count) {
                v.add(-1);
            } else {
                v.add(m.get(queries.get(i)));
            }
        }
        return v;
    }
```

</details>

## Question - 3 : Find the Number of Distinct Colors Among the Balls

<details>
<summary>Python</summary>

```python
def query_results(limit, queries):
    """
    Processes queries about ball colors and efficiently returns the number of distinct colors after each query.

    Args:
        limit: An integer representing the maximum number of colors allowed. (Not used in this implementation)
        queries: A list of lists, where each sublist represents a query with [ball, color].

    Returns:
        A list of integers, where each element represents the number of distinct colors after the corresponding query.
    """

    # gives colBalls[color]
    col_balls = {}
    # gives ballColor[ball]
    ball_color = {}

    res = []

    # Process Queries
    for ball, color in queries:
        # if - ball is not coloured
        if ball not in ball_color:
            ball_color[ball] = color
            col_balls[color] = col_balls.get(color, 0) + 1
        else:
            old_col = ball_color[ball]

            # if this is the last ball of old color
            if col_balls[old_col] == 1:
                del col_balls[old_col]
            else:
                col_balls[old_col] -= 1

            # update the colBalls and ballColor
            ball_color[ball] = color
            col_balls[color] = col_balls.get(color, 0) + 1

        res.append(len(col_balls))

    return res
```

</details>

<details>
<summary>Cpp</summary>

```cpp
class Solution {
public:
    vector<int> queryResults(int limit, vector<vector<int>>& queries) {
        /* gives colBalls[color] */ 
        map<int, int> colBalls;
        /* gives ballColor[ball] */
        map<int, int> ballColor;
        
        vector<int> res;
        
        // Process Queries
        for (int i = 0; i < queries.size(); i++) {
            int ball = queries[i][0];
            int color = queries[i][1];

            // if - ball is not coloured
            if (!ballColor.count(ball)) {
                ballColor[ball] = color;
                colBalls[color]++;
            }
            // else - ball is recolored
            else {
                int oldCol = ballColor[ball];

                // if this is the last ball of old color
                if (colBalls[oldCol] == 1)
                    colBalls.erase(oldCol);
                else
                    colBalls[oldCol]--;
                
                // update the colBalls and ballColor
                ballColor[ball] = color;
                colBalls[color]++;
            }
            res.push_back(colBalls.size());
        }
        return res;
    }
};
```

</details>

<details>
<summary>Java</summary>

```java
public class Solution {

public List<Integer> queryResults(int limit, List<List<Integer>> queries) {
    /* gives colBalls[color] */
    Map<Integer, Integer> colBalls = new HashMap<>();
    /* gives ballColor[ball] */
    Map<Integer, Integer> ballColor = new HashMap<>();

    List<Integer> res = new ArrayList<>();

    // Process Queries
    for (int i = 0; i < queries.size(); i++) {
        int ball = queries.get(i).get(0);
        int color = queries.get(i).get(1);

        // if - ball is not coloured
        if (!ballColor.containsKey(ball)) {
            ballColor.put(ball, color);
            colBalls.put(color, colBalls.getOrDefault(color, 0) + 1);
        } else {
            int oldCol = ballColor.get(ball);

            // if this is the last ball of old color
            if (colBalls.get(oldCol) == 1) {
                colBalls.remove(oldCol);
            } else {
                colBalls.put(oldCol, colBalls.get(oldCol) - 1);
            }

            // update the colBalls and ballColor
            ballColor.put(ball, color);
            colBalls.put(color, colBalls.getOrDefault(color, 0) + 1);
        }
        res.add(colBalls.size());
    }

    return res;
}
}
```

</details>

