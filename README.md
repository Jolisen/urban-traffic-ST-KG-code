# Urban Traffic ST-KG Code

This repository contains the implementation code for constructing and applying the **Urban Traffic Spatio-Temporal Knowledge Graph (ST-KG)** in Neo4j, as described in our research. The code supports four main applications:  
1. **Spatiotemporal analysis of congestion dynamics**  
2. **Traffic speed prediction**  
3. **Intelligent question answering on congestion**  
4. **Tracing the causes of congestion**

---

## Main Requirements

The code has been tested with the following dependencies:
- Fiona==1.8.22
- folium==0.18.0
- GDAL==3.6.2
- geopandas==0.13.2
- shapely==2.0.3
- langchain==0.2.17
- openai==1.65.2
- py2neo==2021.2.4
- pyarrow==17.0.0
- pytorch-lightning==1.9.0
- torch==1.13.1+cu117

## Data

All datasets required to reproduce the results of the four applications in the paper are available at our Zenodo repository:
https://doi.org/10.5281/zenodo.16777726

## Program Structure & Usage

### **1. TRGK Creator Folder**

This folder constructs the Urban Traffic ST-KG in [Neo4j](https://neo4j.com/):

- `road.ipynb` – Builds **road** entities and `touch` relationships between roads.  
- `grid.ipynb` – Builds **grid** entities and `adjacency` relationships between grids.  
- `POI.ipynb` – Builds **POI** entities and `contains` relationships between POIs and grids.  
- `state.ipynb` – Builds **state** entities, `next` relationships between states, and `located_in` relationships to grids.  
- `Cross relation construction.ipynb` – Builds `within` relationships between roads and grids.  
- `h-name in state.ipynb` – Embeds the ID of the grid entity with a `located_in` relationship into each state entity to optimize queries.  
- `TRKG index creater.ipynb` – Creates indexes for the constructed entities.
<p align="center">
  <img src="images/绘图8.png" alt="ST-KG Structure" width="600"><br>
  <em>Structure of the Urban Traffic Spatio-Temporal Knowledge Graph (ST-KG)</em>
</p>


### **2. Congestion Level Assessment Folder**

- `v_acc.ipynb` – Calculates the average vehicle speed within each grid cell.

---
### **3. Spatiotemporal Analysis of Urban Traffic Congestion Dynamics Folder**

- `temporal mode and group.ipynb` – Divides the April 2015 traffic data into three temporal modes, each containing five time groups, and computes the average speed of all grid cells in each time group.  
- `ST transitions plot.ipynb` – Maps the average speed in each time group to the corresponding congestion level and generates the required visualizations.
### **4. Traffic Speed Prediction at the Regional Scale Folder**

- Prediction model files: `SVR.ipynb`, `gcn.py`, `gru.py`, `tgcn.py`, and `SceneGCN.py`.  
- The `out` folder stores the prediction results.  
- The prediction uses 80% of the data as the training set and the remaining 20% as the test set.  
- The **GCN**, **GRU**, **TGCN**, and **SceneGCN** models are all executed through `main.ipynb`.

### **5. Intelligent Question Answering on Traffic Congestion Folder**

- The construction and application code for the LLM-Agent is contained in `LLM-Traffic-agent.ipynb`.  
- To run the notebook, the user must set their own OpenAI API key in the initialization code:

```python
# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o", 
    api_key="your api",
    base_url="your url",
    temperature=0
)
```
### **6. Tracing the Causes of Non-Recurrent Traffic Congestion Folder**

- The `cause.ipynb` notebook builds upon the results from `LLM-Traffic-agent.ipynb` to further compute traffic flow statistics.  
- It also generates relevant visualizations that can be used in combination with external event data to identify the causes of congestion.




