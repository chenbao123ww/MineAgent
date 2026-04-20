Task: I need a portable bluetooth speaker, waterproof, black color, price lower than $30.00.

Skills:[ele 001] Constraint-rich search; [ele 004] Bail on mismatch fast; [gen 002] Iterative refinement.

Step 1:
    Obs: WebShop homepage. 
    Think: [ele 001] Packing all mandatory attributes: portable, bluetooth, waterproof, black, ¡$30. 
    Action: search[portable bluetooth speaker waterproof black under 30]  
Step 2:
    Obs: Results: [1] Waterproof Speaker $28.99; [2] Mini LED Speaker $19.99; [3] 20W Speaker $35.50. 
    Think: [ele 005] Item [1] matches most keywords and price. 
    Action: click[Item 1]  
Step 3:
    Obs: Product page (Item 1). Colors: Blue, Red, Green, Army Green. Black unavailable. 
    Think: [gen 004] Verify Early, Abort Fast: No black option! [ele 004] Bail immediately to find another candidate.  
    Action: click[Back to Search]  
Step 4:
    Obs: Back on search results. 
    Think: [gen 002] Iterative refinement. Item [2] $19.99 is the next best candidate. 
    Action: click[Item 2]  
Step 5:
    Obs: Product page (Item 2). Colors: Black, Blue, Red. Specs: IPX5 Waterproof. 
    Think: [ele 002] Black is available! IPX5 confirms waterproof. Selecting Black. 
    Action: click[Black]  
Step 6:
    Obs: Black selected. Final price: $19.99. 
    Think: [ele 003] $19.99 ¡ $30. All specs (black, portable, waterproof) verified. 
    Action: click[Buy Now]