### **BUILDING AN UNBEATABLE TIC TAC TOE GAME USING MINMAX ALGORITHM**

Originally Written on:- November,2018.

Tic tac toe is a game for two players, X and O, 
who take turns marking the spaces in a 3×3 grid traditionally. 
The player who succeeds in placing three of their marks in a horizontal, vertical, 
or diagonal row wins the game. Tic Tac Toe can be used as a way of teaching artificial intelligence
that deals with the searching of game trees. Now, in the traditional way Tic tac toe can be made to play 
against other players, but let’s try to develop one that is unbeatable . 
In this blog post I will try to explain as much as possible about building an AI tic tac toe game that is unbeatable. 
I have built one using JavaScript, html and css and you can play it using the link that I will provide .

Now, to build an AI version of tic tac toe that is unbeatable we first need to understand something that is known as the Minmax algorithm.
## **MINMAX ALGORITHM**

This algorithm sees a few steps ahead and puts itself in the shoes of its opponent. 
It keeps playing ahead until it reaches a terminal arrangement of the board (terminal state) 
resulting in a tie, a win, or a loss. Once in a terminal state, the AI will assign an arbitrary positive 
score (+10) for a win, a negative score (-10) for a loss, or a neutral score (0) for a tie.
At the same time, the algorithm evaluates the moves that lead to a terminal state based on the players’ turn.
It will choose the move with maximum score when it is the AI’s turn and choose the move with the minimum 
score when it is the human player’s turn. Using this strategy, Minimax avoids losing to the human player. 
You can play the game that I have developed using the link : 
https://codepen.io/soumya1995/pen/PxpgyO

Or

https://s.codepen.io/soumya1995/debug/PxpgyO/ZoMBaKZmJYqk 


Press the replay button to start a new game.
- A Minimax algorithm can be best defined as a recursive function that does the following things:
- return a value if a terminal state is found (+10, 0, -10)
- go through available spots on the board
- call the minimax function on each available spot (recursion)
- evaluate returning values from function calls
- and return the best value

Now, the algorithm that I have used is in my Github profile: The link to the full source codes is: https://github.com/soumyadip1995/tictactoe/tree/master/tictactoe.

Now, the explanation is in a few parts:

Declaration of variables  and starting the AI using JavaScript.
- In the beginning we try to declare all the variables, we define the human player as “O”  and the AI player as ‘X’ and arrange all the winning combinations in an array. We then use a loop to traverse through all the winning combinations .
- Then we use a turn function to check for the squares in the board using squareId for the human players. We check for both  ties and Win. If the AI wins or the game is tied then the game is over.
- We declare all the functions that is necessary to check if the AI player won or the human player won (needed later on in the minmax algorithm).
- In this game we also need to check for the spaces that are available for both the AI player and the human player. Only, the available space needs to be filled in by both the Human player and the AI player.
- Also, the best spaces needs to be checked in order to be filled in by both the human player and the AI player.
- We need to declare the winner in the end which will always be the AI player using the minmax algorithm described below.

### **MINMAX ALGORITHM EXPLANATION**

To explain the minmax algorithm we first need to understand something that is called as game trees. 
In game theory, a game tree is a directed graph whose nodes are positions in a game and whose edges are moves.
Game tree defines the states that our tic tac toe is going to go through. We shall begin with the state that is 
shown below (Fig source: freecodecamp.org).

Now, the recursive function call  of our tic tac toe will look something like this in order for the AI player to win.
(Fig source: freecodecamp.org).


![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQctrJ3dhxbKiM9NmzLfQmkpNfe36poyIFGk-eQsDnAsyeJORQ7)

### **DESCRIPTION**

![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ2uQiYeTa9CPC8Rs-r5yokaa6R0cpknC2oIeddmi9X6WYjAXn9)



In the Github code the minmax function() is working as described below. It is using recursion:
(If you are like me it will take a few reads to completely understand the procedure) Using the picture above, we are checking for
the terminal states above in the 1st function call and then running a loop through each spot.
Here, the algorithm is going to change to the new board by placing the AI player in the 1st empty spot and after that it
is going to call itself with new board and the human player and wait for the function call to return the value.
So, we are on the 1st function call here, now we are going to the 2nd function call while the 1st function call is 
still running and the 2nd one starts. Find the two empty spots,
it is going to check for the terminal states and loop through the empty spots starting with the 1st one and then
it calls the new board by placing the human player in the 1st empty spot and after that it calls itself 
with new board and AI player waits for the function call to return a value. At the 3rd function call 
(still function 2 and 1 are running) the algorithm makes a list of the empty spots. Now, human player wins and 
it is going to return -10 from function 3 and function 4. Goes back to function 1. Now, in function 5 terminal spot
is reached and AI wins (+10 score), so no recursion here. In state 6 however there are two empty spaces .
Algorithm searches for that empty space. The human player can place in any one of the empty spaces 0s.  
If  human puts in the wrong position then function 8 is called from the 7th function call. 
In the 8th  function call the algorithm searches and AI wins. So a score of +10 is given.
In the function 9 however human wins so -10 score. So, in level 2 when new board is called the maximum value is 
chosen functions 3,4,7 and 9. And from the level 1 the minimum is chosen  from function 2, 5 and 6. 
The algorithm returns the highest value so we can safely say that moving the X to the middle of the box is the best 
possible move and the AI will win.

Minmax algorithm will take some time to understand. 
Don’t worry if you haven’t understood everything in the first read.
Just observe the picture of the function call carefully and the description and
I believe you will have a better understanding of how the algorithm works.
