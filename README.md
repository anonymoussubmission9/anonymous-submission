# FSE-paper

This is the public repository for FSE Submission #379: "Towards Better Graph Neural Neural Network-based Fault Localization Through Enhanced Code Representation".

There are two folders, `DepGraph` and `DepGraphCodeChange`. Both of the tecnhniques are included in these folders. 

Put the pkl file in the root directory of the project based on the different methods. Then run `python runtotal.py project_name`. For example, `python runtotal.py Time`. 
All of our Data can be found here: https://drive.google.com/drive/folders/172F5Gv82hC_Qvsb5eQ_XUw1_98gXcRi7?usp=sharing

For cross-project evaluation run `python job_cross.sh`.

The data is Seperated in two folders:
* DepGraph
* DepGraphCodeChange

---------------------------------------

# Updated Results
A comparison of the fault localization techniques can be found in [Table II](Results/Table2.pdf). For each system, we show the technique with the best MFR in bold (the lower the better). DepGraph w/o Code Change shows the result after adopting Dependency-Enhanced Coverage Graph, and DepGraph shows the result of incorporating both Dependency-Enhanced Coverage Graph and Code Change Information.. The number in the parentheses shows the percentage improvement over [Grace](https://dl.acm.org/doi/10.1145/3468264.3468580). The best result is marked in bold.

# AST Node Types Used in Our Analysis

Here is an overview of the specific Abstract Syntax Tree (AST) node types we have utilized in our analysis, focusing on Java code. Our selection of these nodes is driven by their significance in understanding the program's structure, control flow, and data flow. Below is a categorization and rationale for each type of node included in our analysis.

## Control Flow Statements
- **WhileStatement, IfStatement, ForStatement, SwitchStatement**: Essential for dictating the program's execution flow. These nodes often encompass conditions and loops, where faults are likely to occur.

## Method and Constructor Declarations
- **MethodDeclaration, ConstructorDeclaration**: Key to understanding class functionalities, these nodes represent the entry points for execution and often contain core business logic.

## Exception Handling
- **ThrowStatement**: Used for exception handling in Java. Analyzing these nodes is crucial, especially for faults related to exceptions.

## Termination Statements
- **BreakStatement, ContinueStatement, ReturnStatement**: Control the flow within loops and methods. Critical for analyzing how and when loops terminate and how methods return values.

## Local Variable Declarations
- Local variable declarations are pivotal for tracking the state and flow of data within various scopes, such as methods or blocks.

## Statement Expressions
- **StatementExpression**: These expressions, which can include method invocations or assignments, are crucial for understanding the operations performed in the code.

## Parameters and Return Types
- The `parameters` and `return_type` nodes help in analyzing method signatures and their interactions, which is fundamental for understanding method functionalities.

## Miscellaneous Nodes
- **control, SimpleName**: The `control` node likely refers to additional control flow elements, while `SimpleName` (excluded in our analysis) typically represents variable and method names. `SimpleName` is omitted to reduce complexity and focus on more impactful AST elements.

## Rationale for Node Selection
Our selection criteria for these nodes were based on their relevance in portraying the code's logical and structural flow. By focusing on these specific aspects, we aim to capture the critical elements of the code that are most relevant for tasks like fault localization, program comprehension, or static analysis. This approach helps in reducing the complexity of the analysis while ensuring that the essential elements are thoroughly examined.
