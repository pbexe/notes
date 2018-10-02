## Preliminaries
* Up to three slots of content per week
* 25% Coursework Announced Next Week (2 courseworks ???)
* 50% exam 
* 20 lectures

## Outline

* Intro (2 lectures)
* Network types (2 lectures)
* Standards (8 lectures)
* Applications (4 lectures)
* Security (3 lectures)
* Network Modeling (1 Lecture)

## Advice

* Do work outside of lectures
* Read RFCs
    * Example: [Telnet](https://tools.ietf.org/html/rfc854.html)
    * Example:  [Coffe Pot Protocol (April Fools)](https://tools.ietf.org/html/rfc2324) 
* Play with network stuff
* Contact Phillip
    * ReineckeP@cardiff.ac.uk
    * WX/3.10

## Actual material

### Keywords 
| Keyword | Definition |
| -------- | -------- |
| Network | Computers connected so they can communicate with eachother |
| Protocol | Set of rules for communication between entities |

### Content

* You can send a request to a website using the telnet command, the website will return plaintext information
* This is because HTTP is a plaintext protocol
* Wireshark is used to visualise network traffic
* What is a network?
    * Computers connected so they can communicate with eachother

#### Layers

- Decomposition - One protocol per layer
- Abstraction - Interfaces between layers
- Reusability - re-use protocols
![pdntspa](https://i.imgur.com/yHiZ215.png)
> visual aids help learning (anki approved)
did you just draw this?
Yup. Using my laptop 


OSI 7 layer stack:
- **P**lease **D**o **N**ot **T**hrow **S**ausage **P**izza **A**way
- Physical - Physical Connections and bit stream, cables, wireless transmission etc.
- Data Link - Aggregates stream of bits into a frame
- Network - Routing and flow control of packets
- Transport - Splits/joins messages into packets for sending over lower layers. Also handles packet loss and out of order delivery
- Session - Maintains connection between processes
- Presentation - Perty format
- Application - The appplication that the user sees
