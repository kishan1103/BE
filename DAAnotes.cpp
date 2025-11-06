
// You said:
// #include <iostream>
// #include <vector>
// using namespace std;

// int knapsack(int W, vector<int> &wt, vector<int> &val, int n) {
//     // Create DP table
//     vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

//     // Build table dp[][] in bottom-up manner
//     for (int i = 0; i <= n; i++) {
//         for (int w = 0; w <= W; w++) {
//             if (i == 0 || w == 0)
//                 dp[i][w] = 0;  // Base case
//             else if (wt[i - 1] <= w)
//                 dp[i][w] = max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]);
//             else
//                 dp[i][w] = dp[i - 1][w];
//         }
//     }

//     // dp[n][W] contains the maximum value
//     return dp[n][W];
// }

// int main() {
//     int n, W;
//     cout << "Enter number of items: ";
//     cin >> n;

//     vector<int> val(n), wt(n);
//     cout << "Enter value and weight of each item:\n";
//     for (int i = 0; i < n; i++)
//         cin >> val[i] >> wt[i];

//     cout << "Enter capacity of knapsack: ";
//     cin >> W;

//     int maxValue = knapsack(W, wt, val, n);
//     cout << "Maximum value in 0-1 Knapsack = " << maxValue << endl;

//     return 0;
// }

// // Input

// // Number of items: 3
// // Values & weights:
// // 60 10
// // 100 20
// // 120 30
// // Knapsack capacity: 50

// explain me this code
// ChatGPT said:
// Sure 👍 Let’s go through your 0-1 Knapsack problem (Dynamic Programming) code step by step — in a simple and clear way.

// 🧠 Concept: What is the 0-1 Knapsack Problem?
// You have:

// n items, each having:

// value val[i]

// weight wt[i]

// A knapsack (bag) that can carry a maximum weight of W.

// You want to maximize total value such that the total weight ≤ W.

// 👉 In 0-1 Knapsack, each item can either be:

// Included (1) or

// Excluded (0)
// No fractions allowed (unlike fractional knapsack).

// 🧩 Code Explanation
// 1️⃣ Header & setup
// #include <iostream>
// #include <vector>
// using namespace std;
// #include <vector> — We use dynamic 2D arrays for the DP table.

// using namespace std; — So we don’t need to write std:: every time.

// 2️⃣ Function: knapsack()
// int knapsack(int W, vector<int> &wt, vector<int> &val, int n)
// Parameters:

// W: Capacity of knapsack.

// wt: Array of item weights.

// val: Array of item values.

// n: Number of items.

// 3️⃣ Create DP Table
// vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
// We make a 2D DP array of size (n+1) × (W+1)
// dp[i][w] = maximum value possible with first i items and capacity w.

// Initially, all values are 0.

// 4️⃣ Fill the DP table (Bottom-Up)
// for (int i = 0; i <= n; i++) {
//     for (int w = 0; w <= W; w++) {
//         if (i == 0 || w == 0)
//             dp[i][w] = 0;  // Base case
//         else if (wt[i - 1] <= w)
//             dp[i][w] = max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]);
//         else
//             dp[i][w] = dp[i - 1][w];
//     }
// }
// Let’s break this down 👇
// Base case:
// When there are 0 items or 0 capacity, the maximum value = 0.

// If item can fit (wt[i-1] <= w):
// You have 2 choices:

// Include the item → val[i-1] + dp[i-1][w - wt[i-1]]

// Exclude the item → dp[i-1][w]

// Take the maximum of the two.

// If item is too heavy (wt[i-1] > w):
// You can’t include it → so copy value from above:
// dp[i][w] = dp[i-1][w]

// 5️⃣ Return the final result
// return dp[n][W];
// The cell dp[n][W] contains the maximum value possible using n items and capacity W.

// 6️⃣ Main Function
// int main() {
//     int n, W;
//     cout << "Enter number of items: ";
//     cin >> n;

//     vector<int> val(n), wt(n);
//     cout << "Enter value and weight of each item:\n";
//     for (int i = 0; i < n; i++)
//         cin >> val[i] >> wt[i];

//     cout << "Enter capacity of knapsack: ";
//     cin >> W;

//     int maxValue = knapsack(W, wt, val, n);
//     cout << "Maximum value in 0-1 Knapsack = " << maxValue << endl;

//     return 0;
// }
// Here we:

// Take number of items n.

// Take input for val[] and wt[].

// Take capacity W.

// Call the knapsack() function and print result.

// 🧮 Dry Run (Input Example)
// Items = 3
// Values = [60, 100, 120]
// Weights = [10, 20, 30]
// Capacity = 50
// Item	Weight	Value
// 1	10	60
// 2	20	100
// 3	30	120
// Building DP Table:

// After full computation:

// Maximum value = 220

// ✅ Optimal selection:
// Take item 2 (100, 20) and item 3 (120, 30) → total weight = 50, total value = 220.

// 🧾 Output
// Enter number of items: 3
// Enter value and weight of each item:
// 60 10
// 100 20
// 120 30
// Enter capacity of knapsack: 50
// Maximum value in 0-1 Knapsack = 220
// ⚙️ Time and Space Complexity
// Time Complexity: O(n * W)
// → nested loops for n items and W capacity.

// Space Complexity: O(n * W)
// → because of 2D DP table.

// (Can be optimized to O(W) using a 1D array — advanced version.)

// Would you like me to show a step-by-step DP table (matrix) for this example so you can see exactly how dp[][] gets filled?

// You said:
// #include <iostream>
// #include <vector>
// #include <algorithm>
// using namespace std;

// // Structure to represent an item
// struct Item {
//     double value;
//     double weight;
// };

// // Comparator function to sort items by value/weight ratio
// bool cmp(Item a, Item b) {
//     double r1 = a.value / a.weight;
//     double r2 = b.value / b.weight;
//     return r1 > r2; // Descending order
// }

// // Function to solve fractional knapsack
// double fractionalKnapsack(int n, double W, vector<Item> &items) {
//     // Sort items by value/weight ratio
//     sort(items.begin(), items.end(), cmp);

//     double totalValue = 0.0;

//     for (int i = 0; i < n; i++) {
//         if (items[i].weight <= W) {
//             // Take whole item
//             W -= items[i].weight;
//             totalValue += items[i].value;
//         } else {
//             // Take fractional part
//             totalValue += items[i].value * (W / items[i].weight);
//             break; // Knapsack is full
//         }
//     }

//     return totalValue;
// }

// int main() {
//     int n;
//     double W;
//     cout << "Enter number of items: ";
//     cin >> n;
//     vector<Item> items(n);

//     cout << "Enter value and weight of each item:\n";
//     for (int i = 0; i < n; i++) {
//         cin >> items[i].value >> items[i].weight;
//     }

//     cout << "Enter capacity of knapsack: ";
//     cin >> W;

//     double maxValue = fractionalKnapsack(n, W, items);
//     cout << "Maximum value in the knapsack = " << maxValue << endl;

//     return 0;
// }

// // Input

// // Number of items: 3
// // Values & weights:
// // 60 10
// // 100 20
// // 120 30
// // Knapsack capacity: 50

// explain me this code
// ChatGPT said:
// Perfect 👌 — this code is for the Fractional Knapsack Problem (Greedy method).
// Let’s break it down step by step so you understand exactly what it’s doing and why.

// 🧠 Concept: What is Fractional Knapsack?
// You have:

// n items, each having:

// a value

// a weight

// a knapsack (bag) that can carry a maximum weight W.

// You need to maximize total value,
// but — unlike 0-1 knapsack —
// 👉 you can take fractions of items.

// That means if an item is too heavy,
// you can take a part of it proportional to the remaining capacity.

// This is solved efficiently using a greedy approach, not DP.

// 🧩 Code Explanation
// 1️⃣ Include Headers
// #include <iostream>
// #include <vector>
// #include <algorithm>
// using namespace std;
// We include:

// <vector> → for dynamic arrays

// <algorithm> → for sorting items based on ratio

// 2️⃣ Item Structure
// struct Item {
//     double value;
//     double weight;
// };
// We define a structure to hold:

// value of the item

// weight of the item

// 3️⃣ Comparator Function
// bool cmp(Item a, Item b) {
//     double r1 = a.value / a.weight;
//     double r2 = b.value / b.weight;
//     return r1 > r2; // Sort in descending order of value/weight ratio
// }
// This function compares two items based on their value-to-weight ratio.

// 👉 The item with the higher ratio (more value per kg) should come first.
// That’s why we sort in descending order (r1 > r2).

// 4️⃣ Fractional Knapsack Function
// double fractionalKnapsack(int n, double W, vector<Item> &items)
// Step 1: Sort items by ratio
// sort(items.begin(), items.end(), cmp);
// Now, the item with highest value/weight ratio comes first.

// Step 2: Initialize total value
// double totalValue = 0.0;
// This will store the total profit/value of the knapsack.

// Step 3: Traverse through all items
// for (int i = 0; i < n; i++) {
//     if (items[i].weight <= W) {
//         // Take whole item
//         W -= items[i].weight;
//         totalValue += items[i].value;
//     } else {
//         // Take fractional part
//         totalValue += items[i].value * (W / items[i].weight);
//         break; // Knapsack is full
//     }
// }
// 👉 Logic:

// If the item fits fully (items[i].weight <= W):

// Take it completely.

// Subtract its weight from remaining capacity.

// Add its full value.

// Else (the item is heavier than the remaining capacity):

// Take only a fraction equal to (W / weight).

// Add that fraction’s value.

// Then break, since the bag is now full.

// Step 4: Return total value
// return totalValue;
// 5️⃣ Main Function
// int main() {
//     int n;
//     double W;
//     cout << "Enter number of items: ";
//     cin >> n;
//     vector<Item> items(n);

//     cout << "Enter value and weight of each item:\n";
//     for (int i = 0; i < n; i++) {
//         cin >> items[i].value >> items[i].weight;
//     }

//     cout << "Enter capacity of knapsack: ";
//     cin >> W;

//     double maxValue = fractionalKnapsack(n, W, items);
//     cout << "Maximum value in the knapsack = " << maxValue << endl;

//     return 0;
// }
// Takes input for:

// Number of items

// Value and weight of each item

// Capacity of knapsack (W)

// Calls the fractionalKnapsack() function.

// Prints the result.

// 🧮 Dry Run (Example)
// Input:

// 3
// 60 10
// 100 20
// 120 30
// 50
// Item	Value	Weight	Value/Weight
// 1	60	10	6.0
// 2	100	20	5.0
// 3	120	30	4.0
// Sorted (by ratio): Item 1 → Item 2 → Item 3 ✅

// Capacity = 50

// Step	Item	Take	Remaining W	Added Value	Total Value
// 1	Item 1	Full	40	+60	60
// 2	Item 2	Full	20	+100	160
// 3	Item 3	Partial (20/30 = 2/3)	0	+120 × (20/30) = 80	240
// ✅ Final Maximum Value = 240

// 🧾 Output
// Enter number of items: 3
// Enter value and weight of each item:
// 60 10
// 100 20
// 120 30
// Enter capacity of knapsack: 50
// Maximum value in the knapsack = 240
// ⚙️ Complexity Analysis
// Type	Complexity
// Time	O(n log n) (due to sorting) + O(n) (for loop) ⇒ O(n log n)
// Space	O(1) (no extra memory used except small variables)
// 🧩 Summary
// Feature	0/1 Knapsack	Fractional Knapsack
// Approach	Dynamic Programming	Greedy
// Fraction Allowed?	❌ No	✅ Yes
// Time Complexity	O(n × W)	O(n log n)
// Data Type	int	double
// Example Result	220	240
// Would you like me to show how to visualize the fractional knapsack selection step-by-step (like with percentage of each item taken)?

// You said:
// #include <iostream>
// #include <vector>
// #include <algorithm>
// using namespace std;

// // Function to print the board
// void printBoard(const vector<vector<int>>& board, int N) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++)
//             cout << board[i][j] << " ";
//         cout << endl;
//     }
// }

// // Function to check if placing a queen at (row, col) is safe
// bool isSafe(const vector<vector<int>>& board, int row, int col, int N) {
//     // Check same column
//     for (int i = 0; i < row; i++)
//         if (board[i][col] == 1) return false;

//     // Check upper-left diagonal
//     for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
//         if (board[i][j] == 1) return false;

//     // Check upper-right diagonal
//     for (int i = row - 1, j = col + 1; i >= 0 && j < N; i--, j++)
//         if (board[i][j] == 1) return false;

//     return true;
// }

// // Backtracking function to place queens
// bool solveNQueens(vector<vector<int>>& board, int row, int N) {
//     if (row >= N) return true; // All queens placed

//     // Skip row if queen already placed
//     if (find(board[row].begin(), board[row].end(), 1) != board[row].end())
//         return solveNQueens(board, row + 1, N);

//     for (int col = 0; col < N; col++) {
//         if (board[row][col] == 0 && isSafe(board, row, col, N)) {
//             board[row][col] = 1; // Place queen
//             if (solveNQueens(board, row + 1, N))
//                 return true;
//             board[row][col] = 0; // Backtrack
//         }
//     }

//     return false; // No valid position
// }

// int main() {
//     int N;
//     cout << "Enter size of board (N): ";
//     cin >> N;

//     vector<vector<int>> board(N, vector<int>(N, 0));

//     int firstRow, firstCol;
//     cout << "Enter position of first queen (row and column, 0-based index): ";
//     cin >> firstRow >> firstCol;

//     if (firstRow >= N || firstCol >= N || firstRow < 0 || firstCol < 0) {
//         cout << "Invalid position!" << endl;
//         return 0;
//     }

//     board[firstRow][firstCol] = 1; // Place first queen

//     // Solve remaining queens starting from row 0
//     if (solveNQueens(board, 0, N)) {
//         cout << "\nN-Queens solution:\n";
//         printBoard(board, N);
//     } else {
//         cout << "No solution exists with the first queen at the given position." << endl;
//     }

//     return 0;
// }

// // Input :
// // Size of board: 4
// // Position of first queen: 0 1

// //size of board: 8
// //position of first queen: 0 0

// explain this code 
// ChatGPT said:
// Great — this is a C++ implementation of the N-Queens problem using backtracking,
// but with a custom twist 👉 you can fix the position of the first queen manually.

// Let’s go through the code step by step to make everything crystal clear 👇

// 🧠 What is the N-Queens Problem?
// You have an N × N chessboard, and you must place N queens such that:

// No two queens attack each other.

// That means:

// No two queens share the same row.

// No two queens share the same column.

// No two queens share the same diagonal.

// 🟡 In this code, you can choose where the first queen starts, and then the program will try to find a valid configuration for the rest.

// 🧩 Code Explanation
// 1️⃣ Include Headers and Namespace
// #include <iostream>
// #include <vector>
// #include <algorithm>
// using namespace std;
// <vector> → used for dynamic 2D array (board).

// <algorithm> → used for the find() function.

// using namespace std; to avoid typing std::.

// 2️⃣ Function: printBoard
// void printBoard(const vector<vector<int>>& board, int N) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++)
//             cout << board[i][j] << " ";
//         cout << endl;
//     }
// }
// ✅ Purpose:
// Prints the entire chessboard.

// 1 → Queen placed.

// 0 → Empty cell.

// Example for N=4:

// 0 1 0 0
// 0 0 0 1
// 1 0 0 0
// 0 0 1 0
// 3️⃣ Function: isSafe
// bool isSafe(const vector<vector<int>>& board, int row, int col, int N)
// ✅ Purpose:
// Checks whether a queen can be placed at position (row, col) safely.

// Checks performed:
// 1️⃣ Same column:

// for (int i = 0; i < row; i++)
//     if (board[i][col] == 1) return false;
// → No other queen should be in the same column above.

// 2️⃣ Upper-left diagonal:

// for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
//     if (board[i][j] == 1) return false;
// → Checks the diagonal going up-left.

// 3️⃣ Upper-right diagonal:

// for (int i = row - 1, j = col + 1; i >= 0 && j < N; i--, j++)
//     if (board[i][j] == 1) return false;
// → Checks the diagonal going up-right.

// If all checks pass:

// return true;
// ✅ Safe to place the queen here.

// 4️⃣ Function: solveNQueens
// bool solveNQueens(vector<vector<int>>& board, int row, int N)
// ✅ Purpose:
// Places queens row by row using backtracking.

// Base Case
// if (row >= N) return true; // All queens are placed
// If all rows are filled → success!

// Skip row if already filled:
// if (find(board[row].begin(), board[row].end(), 1) != board[row].end())
//     return solveNQueens(board, row + 1, N);
// 👉 This is the twist —
// If you already placed a queen in that row (e.g. the first one given by user), skip to next row.

// Try placing queens in each column
// for (int col = 0; col < N; col++) {
//     if (board[row][col] == 0 && isSafe(board, row, col, N)) {
//         board[row][col] = 1; // Place queen
//         if (solveNQueens(board, row + 1, N))
//             return true;
//         board[row][col] = 0; // Backtrack
//     }
// }
// Explanation:

// For each column in the current row:

// Check if it’s safe to place.

// If yes:

// Place queen (board[row][col] = 1)

// Recursively solve next row.

// If that leads to a solution → return true.

// Otherwise → backtrack (remove queen, try next column).

// If no column works:
// return false;
// → Backtrack to previous row.

// 5️⃣ Main Function
// int main() {
//     int N;
//     cout << "Enter size of board (N): ";
//     cin >> N;

//     vector<vector<int>> board(N, vector<int>(N, 0));

//     int firstRow, firstCol;
//     cout << "Enter position of first queen (row and column, 0-based index): ";
//     cin >> firstRow >> firstCol;

//     if (firstRow >= N || firstCol >= N || firstRow < 0 || firstCol < 0) {
//         cout << "Invalid position!" << endl;
//         return 0;
//     }

//     board[firstRow][firstCol] = 1; // Place first queen

//     if (solveNQueens(board, 0, N)) {
//         cout << "\nN-Queens solution:\n";
//         printBoard(board, N);
//     } else {
//         cout << "No solution exists with the first queen at the given position." << endl;
//     }

//     return 0;
// }
// 🧮 Example 1
// Input:
// Size of board: 4
// Position of first queen: 0 1
// ✅ Output:

// N-Queens solution:
// 0 1 0 0
// 0 0 0 1
// 1 0 0 0
// 0 0 1 0
// 🧮 Example 2
// Input:
// Size of board: 8
// Position of first queen: 0 0
// ✅ Output:

// 1 0 0 0 0 0 0 0
// 0 0 0 0 1 0 0 0
// 0 0 0 0 0 0 0 1
// 0 0 0 0 0 1 0 0
// 0 0 1 0 0 0 0 0
// 0 0 0 0 0 0 1 0
// 0 1 0 0 0 0 0 0
// 0 0 0 1 0 0 0 0
// ⚙️ Complexity Analysis
// Type	Explanation	Complexity
// Time	Worst case: try placing queens in all columns for each row	O(N!)
// Space	2D board + recursion stack	O(N²)
// ✅ Summary
// Step	Function	Purpose
// 1	isSafe()	Checks if queen placement is safe
// 2	solveNQueens()	Recursively tries to place queens row by row
// 3	printBoard()	Prints final configuration
// 4	main()	Takes input, initializes board, calls solver
// Would you like me to show a visual step-by-step backtracking process (how it places and removes queens row by row for N=4)? It helps to clearly see how the recursion works.

// You said:
// #include <iostream>
// using namespace std;

// int stepCount = 0; // global variable to count steps

// // Recursive Fibonacci function with step counter
// int fibRecursive(int n) {
//     stepCount++;
//     if (n == 0 || n == 1)
//         return n;
//     return fibRecursive(n - 1) + fibRecursive(n - 2);
// }

// // Iterative (Dynamic Programming) Fibonacci with step counter
// int fibIterative(int n) {
//     stepCount = 0;
//     if (n == 0)
//         return 0;
//     if (n == 1)
//         return 1;

//     int a = 0, b = 1, c;
//     stepCount = 2; // initializing a and b

//     for (int i = 2; i <= n; i++) {
//         c = a + b;
//         a = b;
//         b = c;
//         stepCount++; // each loop iteration counts as one step
//     }

//     return b;
// }

// int main() {
//     int n;
//     cout << "Enter n: ";
//     cin >> n;

//     // Recursive method
//     stepCount = 0;
//     int resultRec = fibRecursive(n);
//     cout << "\n[Fibonacci using Recursion]\n";
//     cout << "F(" << n << ") = " << resultRec << endl;
//     cout << "Step count (Recursive): " << stepCount << endl;

//     // Iterative method
//     int resultIter = fibIterative(n);
//     cout << "\n[Fibonacci using Iteration]\n";
//     cout << "F(" << n << ") = " << resultIter << endl;
//     cout << "Step count (Iterative): " << stepCount << endl;

//     return 0;
// }

// explain this code

// ChatGPT said:
// Perfect 👌 — this is a very nice comparison program showing Recursive vs Iterative Fibonacci approaches with step counting (to compare efficiency).

// Let’s break this down clearly 👇

// 🧠 What is the Fibonacci Sequence?
// The Fibonacci sequence is:

// 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
// Each term after the first two is the sum of the previous two:

// F(n) = F(n-1) + F(n-2)
// with:

// F(0) = 0
// F(1) = 1
// ⚙️ Program Overview
// The program calculates F(n) (the nth Fibonacci number) in two ways:

// Recursive → uses repeated function calls

// Iterative (Dynamic Programming) → uses a loop

// It also counts how many steps (function calls or loop iterations) each method takes.

// 🧩 Detailed Code Explanation
// 1️⃣ Global Step Counter
// int stepCount = 0; // global variable to count steps
// Used to count how many steps the algorithm takes.

// Shared by both recursive and iterative functions.

// 2️⃣ Recursive Fibonacci Function
// int fibRecursive(int n) {
//     stepCount++;
//     if (n == 0 || n == 1)
//         return n;
//     return fibRecursive(n - 1) + fibRecursive(n - 2);
// }
// 🧩 Working:
// Every time fibRecursive() is called → increment stepCount.

// Base case:
// If n == 0 or n == 1, just return n (no further recursion).

// Recursive case:
// Return sum of previous two Fibonacci numbers:

// fibRecursive(n-1) + fibRecursive(n-2)
// ✅ So it makes two recursive calls for each n > 1.

// 🧮 Example for n = 4
// Call Tree:

// fib(4)
//  ├─ fib(3)
//  │   ├─ fib(2)
//  │   │   ├─ fib(1)
//  │   │   └─ fib(0)
//  │   └─ fib(1)
//  └─ fib(2)
//      ├─ fib(1)
//      └─ fib(0)
// ➡️ stepCount = number of calls made = 9
// (you can count each fib() call as one step).

// ❗Disadvantage:
// Recomputes values again and again.

// Time complexity: O(2^n) — exponential!
// (very slow for large n)

// 3️⃣ Iterative (Dynamic Programming) Version
// int fibIterative(int n) {
//     stepCount = 0;
//     if (n == 0)
//         return 0;
//     if (n == 1)
//         return 1;

//     int a = 0, b = 1, c;
//     stepCount = 2; // initializing a and b

//     for (int i = 2; i <= n; i++) {
//         c = a + b;
//         a = b;
//         b = c;
//         stepCount++; // each loop iteration counts as one step
//     }

//     return b;
// }
// 🧩 Working:
// Base cases:

// F(0) = 0

// F(1) = 1

// Initialize:

// a = 0, b = 1
// These hold the last two Fibonacci values.

// For each iteration:

// c = a + b   → next Fibonacci number
// a = b
// b = c
// Repeat until i = n.

// ✅ Time Complexity: O(n)
// ✅ Space Complexity: O(1)

// 🧮 Example for n = 4
// a = 0, b = 1
// i=2 → c=1 → a=1, b=1
// i=3 → c=2 → a=1, b=2
// i=4 → c=3 → a=2, b=3
// Output: F(4) = 3
// Step count = 2 (init) + 3 (loop) = 5

// 4️⃣ Main Function
// int main() {
//     int n;
//     cout << "Enter n: ";
//     cin >> n;

//     // Recursive method
//     stepCount = 0;
//     int resultRec = fibRecursive(n);
//     cout << "\n[Fibonacci using Recursion]\n";
//     cout << "F(" << n << ") = " << resultRec << endl;
//     cout << "Step count (Recursive): " << stepCount << endl;

//     // Iterative method
//     int resultIter = fibIterative(n);
//     cout << "\n[Fibonacci using Iteration]\n";
//     cout << "F(" << n << ") = " << resultIter << endl;
//     cout << "Step count (Iterative): " << stepCount << endl;

//     return 0;
// }
// 🧩 What happens here:
// Takes input n.

// Calls recursive version → shows result + step count.

// Calls iterative version → shows result + step count.

// Prints both to compare.

// 🧮 Example Run
// Input:
// Enter n: 5
// Recursive:
// F(5) = 5
// Step count (Recursive): 15
// Iterative:
// F(5) = 5
// Step count (Iterative): 7
// ✅ Both give the same Fibonacci number (5),
// but the recursive one takes many more steps.

// 📊 Performance Comparison
// Method	Time Complexity	Space	Step Count (approx)	Approach
// Recursive	O(2ⁿ)	O(n) (call stack)	Explodes fast as n increases	Divide & Conquer
// Iterative	O(n)	O(1)	Increases linearly	Dynamic Programming
// 🧩 Example Summary
// n	Recursive Steps	Iterative Steps	Result
// 3	5	4	2
// 5	15	7	5
// 10	177	12	55
// 🧠 Key Takeaway:
// Recursive version is simple but inefficient.

// Iterative (DP) version is fast and optimized.

// Step count clearly shows the difference in performance.

// Would you like me to show the call tree diagram for the recursive Fibonacci (e.g., for n=5) so you can visually see how it explodes in size?

// You said:
// #include <iostream>
// #include <vector>
// #include <algorithm>
// using namespace std;

// struct Job {
//     int id;       // Job ID
//     int deadline; // Deadline of job
//     int profit;   // Profit if job is done before or on deadline
// };

// // Comparator to sort jobs by descending profit
// bool compare(Job a, Job b) {
//     return a.profit > b.profit;
// }

// int main() {
//     int n;
//     cout << "Enter number of jobs: ";
//     cin >> n;

//     vector<Job> jobs(n);
//     cout << "Enter job details (id deadline profit):\n";
//     for (int i = 0; i < n; i++) {
//         cin >> jobs[i].id >> jobs[i].deadline >> jobs[i].profit;
//     }

//     // Step 1: Sort jobs by profit in decreasing order
//     sort(jobs.begin(), jobs.end(), compare);

//     // Step 2: Find maximum deadline
//     int maxDeadline = 0;
//     for (auto &job : jobs)
//         maxDeadline = max(maxDeadline, job.deadline);

//     // Step 3: Create a time slot array
//     vector<int> slot(maxDeadline + 1, -1); // -1 means empty slot

//     int countJobs = 0, totalProfit = 0;

//     // Step 4: Assign jobs to slots
//     for (auto &job : jobs) {
//         // Find a free slot from job.deadline down to 1
//         for (int j = job.deadline; j > 0; j--) {
//             if (slot[j] == -1) {
//                 slot[j] = job.id;
//                 countJobs++;
//                 totalProfit += job.profit;
//                 break;
//             }
//         }
//     }

//     // Step 5: Output result
//     cout << "\nJobs scheduled: ";
//     for (int i = 1; i <= maxDeadline; i++) {
//         if (slot[i] != -1)
//             cout << "J" << slot[i] << " ";
//     }
//     cout << "\nTotal number of jobs done: " << countJobs;
//     cout << "\nTotal profit: " << totalProfit << endl;

//     return 0;
// }

// explain me this code

// ChatGPT said:
// Perfect — let’s go through your Job Sequencing with Deadlines (Greedy Method) program line by line 👇

// 🧩 Purpose of the Program
// This program schedules jobs to maximize profit, where:

// Each job takes 1 unit of time.

// Each job has a deadline (must be completed before or on that deadline).

// If multiple jobs compete for the same time slot, we select the most profitable one first (Greedy choice).

// 🧱 Code Breakdown
// #include <iostream>
// #include <vector>
// #include <algorithm>
// using namespace std;
// These are standard header files for input/output and STL containers (vector, sort, etc.).

// 🧩 Structure for Job
// struct Job {
//     int id;       // Job ID
//     int deadline; // Deadline of job
//     int profit;   // Profit if job is done before or on deadline
// };
// Defines a custom data structure Job.

// Each job has:

// id — Job number

// deadline — The time before which it must be completed

// profit — Reward for completing it on time.

// 🔄 Comparator Function
// bool compare(Job a, Job b) {
//     return a.profit > b.profit;
// }
// This function is used to sort jobs in descending order of profit, so the most profitable job is considered first.

// 📥 Input Section
// int n;
// cout << "Enter number of jobs: ";
// cin >> n;

// vector<Job> jobs(n);
// cout << "Enter job details (id deadline profit):\n";
// for (int i = 0; i < n; i++) {
//     cin >> jobs[i].id >> jobs[i].deadline >> jobs[i].profit;
// }
// User inputs number of jobs (n).

// For each job, enters:

// Job ID, Deadline, and Profit.

// These are stored in a vector of Job.

// 🧠 Step 1: Sort Jobs by Profit
// sort(jobs.begin(), jobs.end(), compare);
// Sorts all jobs so that the most profitable job comes first.

// This helps the greedy algorithm always pick the best possible job next.

// 🕓 Step 2: Find Maximum Deadline
// int maxDeadline = 0;
// for (auto &job : jobs)
//     maxDeadline = max(maxDeadline, job.deadline);
// Finds the maximum deadline among all jobs.

// This helps us know how many time slots we need to consider (from 1 to maxDeadline).

// 📆 Step 3: Create a Time Slot Array
// vector<int> slot(maxDeadline + 1, -1); // -1 means empty slot
// Each index slot[i] represents a time slot (1-based indexing).

// -1 indicates that the slot is empty (no job assigned yet).

// ⚙️ Step 4: Assign Jobs to Slots (Greedy Approach)
// int countJobs = 0, totalProfit = 0;

// for (auto &job : jobs) {
//     for (int j = job.deadline; j > 0; j--) {
//         if (slot[j] == -1) {
//             slot[j] = job.id;
//             countJobs++;
//             totalProfit += job.profit;
//             break;
//         }
//     }
// }
// Here’s what happens:

// For each job (in sorted order of profit):

// Try to schedule it in the latest free slot before its deadline.

// This ensures we can still keep earlier slots free for other jobs with earlier deadlines.

// If the slot is empty (-1), assign the job there.

// Increase:

// countJobs (number of jobs done)

// totalProfit (sum of profits)

// ⚡ This is the core greedy step.

// 🧾 Step 5: Display the Results
// cout << "\nJobs scheduled: ";
// for (int i = 1; i <= maxDeadline; i++) {
//     if (slot[i] != -1)
//         cout << "J" << slot[i] << " ";
// }
// cout << "\nTotal number of jobs done: " << countJobs;
// cout << "\nTotal profit: " << totalProfit << endl;
// Prints which jobs are selected (J1, J2, etc.).

// Prints total number of jobs and total profit earned.

// 🧮 Example Input
// Enter number of jobs: 5
// Enter job details (id deadline profit):
// 1 2 100
// 2 1 19
// 3 2 27
// 4 1 25
// 5 3 15
// ✅ Sorted by Profit (Descending):
// Job ID	Deadline	Profit
// 1	2	100
// 3	2	27
// 4	1	25
// 2	1	19
// 5	3	15
// ⚙️ Scheduling Process
// J1 → slot 2 (profit 100)

// J3 → slot 1 (profit 27)

// J4 → skip (both slots full)

// J5 → slot 3 (profit 15)

// ✅ Total Jobs Done: 3
// ✅ Total Profit: 142
// ✅ Jobs Scheduled: J3 J1 J5

// 🧠 Algorithm Type
// Greedy Algorithm (locally optimal choice → globally optimal result)

// Time Complexity: O(n log n + n * m)

// Sorting → O(n log n)

// Slot assignment → O(n * maxDeadline