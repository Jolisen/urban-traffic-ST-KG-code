# Urban Traffic ST-KG Code

This repository contains the implementation code for **constructing and applying the Urban Traffic Spatio-Temporal Knowledge Graph (ST-KG) in Neo4j**, as described in our research.

Specifically, the repository provides:
- The complete workflow for **Urban Traffic ST-KG construction**, including entity modeling and relationship generation
- Implementations of **four core applications** based on the constructed ST-KG:
  1. **Spatiotemporal analysis of congestion dynamics**
  2. **traffic Flow Speed prediction**
  3. **Intelligent question answering on traffic congestion**
  4. **Tracing and analysis of congestion causes**

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

All datasets required to **construct the Urban Traffic ST-KG** and to **reproduce the results of the four applications** reported in the paper are publicly available in our **Zenodo repository**:

https://doi.org/10.5281/zenodo.16777726

## Program Structure & Usage
### Step 1. TRKG Creator Folder
This step constructs the **Urban Traffic Spatio-Temporal Knowledge Graph (ST-KG)** in **Neo4j**, including entity creation and relationship construction across multiple geographic and temporal layers.
#### Notebooks and Data Description

- **`grid.ipynb`**  
  Constructs **grid entities** and **`adjacency` relationships** between neighboring grids.  
  - `dz_honeycomb_125.shp`: A regular hexagonal grid covering **Shanghai**, with a side length of **125 meters**.  
  - `honeybuffer_125.shp`: A buffered version of `dz_honeycomb_125.shp`, slightlyly larger in extent, used to accurately identify adjacency relationships between grid entities.

- **`road.ipynb`**  
  Constructs **road entities** and **`touch` relationships** between roads.  
  - `split_result.shp`: A preprocessed and segmented **Shanghai urban road network**, where roads have been cleaned and split at intersections.

- **`POI.ipynb`**  
  Constructs **POI entities** and **`contains` relationships** between POIs and grid entities.  
  - `dz_honeycomb_125.shp`: The hexagonal grid used as the spatial container.  
  - `POI/`: A folder storing multiple categories of **Points of Interest (POIs)** in Shanghai.

- **`state.ipynb`**  
  Constructs **state entities**, **`next` relationships** between consecutive states, and **`located_in` relationships** between state entities and grid entities.  
  - `dz_honeycomb_125.shp`: Used to spatially locate states within grids.  
  - `split_result.shp`: Used to associate trajectory states with road entities.  
  - `Taxi_raw_data/`: A folder containing **raw taxi trajectory data** for Shanghai, covering **all vehicle trajectories in April 2015**.  
    - Two sample taxi trajectory files are provided. Additional data are available upon request from the authors.

- **`Cross relation construction.ipynb`**  
  Constructs **`within` relationships** between **road entities** and **grid entities**.  
  - `split_result.shp`: Road network data.  
  - `dz_honeycomb_125.shp`: Hexagonal grid data.  
  - `honeybuffer_125.shp`: Buffered grid data to ensure robust spatial containment detection.

- **`h-name in state.ipynb`**  
  Embeds the **ID of the grid entity** (linked via the `located_in` relationship) directly into each **state entity**, in order to **optimize spatiotemporal query performance** in Neo4j.

- **`TRKG index creater.ipynb`**  
  Creates **indexes** for key entities and attributes in Neo4j to improve query efficiency.

#### Auxiliary Files

- **`folder_name_list.txt`**  
  A list of **taxi IDs**, used to iterate through taxi trajectory folders when constructing state entities.

#### Output

The result of **Step 1** is a fully constructed **Urban Traffic ST-KG** in Neo4j, integrating:

- Road, grid, POI, and state entities  
- Spatial relationships: `adjacency`, `contains`, `within`, `located_in`  
- Temporal relationships: `next`

#### ST-KG Structure

![Structure of the Urban Traffic Spatio-Temporal Knowledge Graph (ST-KG)](images/绘图8.png)

---
### Step 2. Congestion Level Assessment Folder
Based on the Urban Traffic ST-KG constructed in **Step 1**, this step evaluates traffic congestion levels by aggregating and analyzing vehicle speed information at the grid scale.
#### Notebooks and Description

- **`v_acc.ipynb`**  
  Calculates the **average vehicle speed within each grid cell** and performs **vehicle speed level classification** to characterize congestion intensity, as illustrated in the figure below.  
  - `honeycomb_cache.pkl`: Extracted from the constructed ST-KG, storing the **`within` relationships between grid entities and road entities** to accelerate query and computation efficiency.  
  - `roadcrs.txt`: Stores the **coordinate reference system (CRS)** information of the road data to ensure spatial consistency during speed calculation.

#### Speed-Level Visualization

![Vehicle speed classification results](images/绘图9.png)

---
### Step 3. Spatiotemporal Analysis of Urban Traffic Congestion Dynamics Folder
Based on the Urban Traffic ST-KG constructed in **Step 1** and the congestion level assessment results obtained in **Step 2**, this step analyzes the **spatiotemporal dynamics of urban traffic congestion** by organizing traffic data into temporal modes and time groups.
#### Notebooks and Description

- **`temporal mode and group.ipynb`**  
  Divides the **April 2015 traffic data** into **three temporal modes**, each consisting of **five time groups**, and computes the **average vehicle speed of all grid cells** within each time group to characterize temporal congestion patterns.

- **`ST transitions plot.ipynb`**  
  Maps the **average vehicle speed** in each time group to the corresponding **congestion level**, and generates the required **spatiotemporal transition visualizations** of urban traffic congestion.

---
### Step 4. traffic Flow Speed Prediction at the Regional Scale Folder

Based on the spatiotemporal traffic Flow Speed information extracted in **Step 3**, this step performs **grid-level traffic Flow Speed prediction** for the **Huangpu District of Shanghai**.

Specifically, Step 3 provides the **average traffic Flow Speed for each grid cell in each time group**. Building upon these results, this step constructs multiple feature matrices and adjacency information to support regional-scale traffic Flow Speed prediction.

#### Data Preparation

All data required for prediction are stored in the **`prediction_data/`** folder.

- **Feature matrix construction (Traffic Flow Speed features)**  
  Based on the scripts in the `data_process` folder:  
  - `data_manage.ipynb` and `interpolation.ipynb` generate **`feature_matrix_X.csv`**, which serves as the **feature matrix for Traffic Flow Speed prediction**.

- **Adjacency matrix construction**  
  Based on the scripts in the `data_process` folder:  
  - `data_manage2.ipynb` and `adj_create.ipynb` generate **`adj_matrix.csv`**, representing the **road-based adjacency matrix between grid cells**.

- **Static and dynamic attribute matrix construction**  
  Based on the scripts in the `data_process` folder:  
  - `POI_class_num.ipynb` generates the **`POI/`** directory, which stores the **counts of different POI categories within each grid cell** (static attributes).  
  - `precipitation.ipynb` generates the **`precipitation/`** directory, which stores **precipitation station information within the Huangpu District** (dynamic attributes).


#### Prediction Models and Execution

- **Prediction model files**  
  - `SVR.ipynb`  
  - `gcn.py`  
  - `gru.py`  
  - `tgcn.py`  
  - `SceneGCN.py`

- **`main.ipynb`**  
  Executes the **GCN**, **GRU**, **TGCN**, and **SceneGCN** models using the prepared prediction data.

- **`out/`**  
  Stores the **Traffic Flow Speed prediction results** of different models.

- **Training and testing split**  
  - 80% of the data are used as the **training set**  
  - 20% of the data are used as the **test set**

#### Evaluation Metrics

The prediction performance is evaluated using the following metrics:

- Root Mean Square Error (RMSE)  
- Mean Absolute Error (MAE)  
- Accuracy (ACC)  
- Coefficient of Determination (R²)  
- Explained Variance (Explained Var)

These metrics are computed as:

```python
rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pred_tensor, gt_tensor)).item()
mae = torchmetrics.functional.mean_absolute_error(pred_tensor, gt_tensor).item()
acc = utils.metrics.accuracy(pred_tensor, gt_tensor).item()
r2 = utils.metrics.r2(pred_tensor, gt_tensor).item()
expl_var = utils.metrics.explained_variance(pred_tensor, gt_tensor).item()
```

---
### Step 5. Intelligent Question Answering on Traffic Congestion Folder

Based on the **Urban Traffic ST-KG constructed in Step 1** and the **congestion assessment results obtained in Step 2**, this step implements an **LLM-Agent–based intelligent question answering system** for traffic congestion analysis.

The LLM-Agent interacts with the Urban Traffic ST-KG to support semantic reasoning and query answering, and the final results are visualized through dedicated visualization utilities.

#### Notebooks and Description

- **`LLM-Traffic-agent.ipynb`**  
  Implements the **construction and application of the LLM-Agent**, enabling intelligent question answering on traffic congestion by querying and reasoning over the Urban Traffic ST-KG.

#### Utilities

- **`utils/congestion_track.py`**  
  Performs **map matching** between **state entities** and **road entities**, enabling road-level grounding of congestion-related states for downstream question answering and analysis.

- **`utils/visualization.py`**  
  Generates **visualized results** for the question answering outputs, facilitating intuitive interpretation of congestion-related queries and reasoning results.

#### Configuration

To run the LLM-Agent notebook, users need to configure their own **OpenAI API key** in the initialization code:

```python
# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key="your api",
    base_url="your url",
    temperature=0
)
```

---
### Step 6. Tracing the Causes of Non-Recurrent Traffic Congestion Folder

Based on the **intelligent question answering results obtained in Step 5**, this step further traces the **causes of non-recurrent traffic congestion** by integrating traffic flow statistics with contextual information.

#### Notebooks and Description

- **`cause.ipynb`**  
  Builds upon the outputs of `LLM-Traffic-agent.ipynb` to further compute **traffic flow statistics**, and generates corresponding **visualizations**.  
  These results can be combined with **external event data** (e.g., weather conditions, special events, or incidents) to identify and analyze the underlying causes of non-recurrent traffic congestion.


## Naming Notes

- In the paper, the *grid* is hexagonal in shape; therefore, it is referred to as **honeycomb** in the dataset.  
- The *state* in the paper is derived from mapped trajectory points and is directly referred to as **trajectory_point** in the dataset.



