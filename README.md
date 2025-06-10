
# System Primitives
The system is composed from two fundemental components: events and containers. Events are the data type [...]. Events need a _bus_ to communicate [...], but they need _isolation_ to prevent miscommunication during parallel execution / threads [...]. Containers ensure _isolation_ of state to the components contained within it during it's lifecycle. Containers are repsonsible for synchronizing data processing to prevent race conditions, duplicate orders, improper access of state using _barriers_ [...]. 

The demands of the system necassitate some innate complexity in our event bus, but this was the lowest level we could put it, which was the right move imo. 



- docs/workflows.md
- strategies have pre-state filters, risk, when needed, is a post state meta-filter (e.g, strategy is losing too much, switch)
- different events can be created for different paradigms, i.e RL training 
- system is basically a configurable container and event system with trading specific logic built around it
- topology effects thinking process, strategy generating signals before portfolio 





main.py - take cli's to point to config, pass any additional args (e.g, --bars 100, --validate, --dataset test)
src/core:
      - events system - not basic, but the heart of the system.
	  - containers - 


 Modern Topology Structure:
  Root Container
  ├── symbol_timeframe containers (one per symbol/timeframe)
  │   └── Streams BARs to root event bus
  ├── feature container (single container, created dynamically from strategies field)
  │   ├── Receives BARs from all symbol_timeframes
  │   ├── Computes all features
  │   ├── Calls strategies
  │   └── Publishes signals (with metadata including price) to root bus
  ├── portfolio containers (one per portfolio)
  │   ├── Subscribes to signals from specific strategies
  │   ├── Receives signals with price metadata
  │   ├── Calls stateless risk functions
  │   ├── Creates orders if approved
  │   └── Publishes orders to root bus
  └── execution container (single container)
      ├── Receives orders from root bus
      ├── Applies slippage/commission (functional)
      └── Publishes fills to root bus (filtered by container_id)
