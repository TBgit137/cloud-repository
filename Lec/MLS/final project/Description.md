# Final Project: Federated Edge Intelligence System

Design and implement a complete federated learning system for heterogeneous edge devices. Build a real or simulated federated learning system that works under realistic edge constraints.

## Project Overview

Your final project is to design and implement a complete federated learning system that operates across heterogeneous edge devices. The system should show how machine learning can be trained and deployed when data is distributed across many devices and cannot simply be centralized.

Your project should integrate key course ideas: federated learning, framework-level optimization, hardware-aware design, model compression, privacy preservation, and complete system implementation.

## Project Options

### Option 1: Federated Wildlife Conservation

**Goal:** Train wildlife detection models across camera traps, acoustic sensors, and ranger devices without sharing raw images, audio, or sensitive location data.

**Participants:** Camera traps, acoustic nodes, ranger phones, park-zone gateways, conservation cloud server.

**Challenges:** Intermittent connectivity, battery-powered sensors, non-IID data across habitats, location-sensitive conservation data.

**Requirements:** Animal/event classification, hierarchical federated learning, secure aggregation, model compression for low-power devices.

### Option 2: Federated Smart Agriculture

**Goal:** Build a crop disease or crop-stress detection system where farms collaboratively train models without exposing farm images, sensor readings, or yield-sensitive information.

**Participants:** Farmer phones, greenhouse cameras, soil sensors, farm gateways, agricultural cooperative server.

**Challenges:** Rural connectivity, different crop types, different climates, non-IID farm data, limited edge hardware.

**Requirements:** Crop disease or stress prediction, device-aware scheduling, differential privacy, compressed mobile or gateway model.

### Option 3: Federated Personalized Education

**Goal:** Design a privacy-preserving learning analytics system where student devices train models to predict learning needs without centralizing individual learning histories.

**Participants:** Student tablets or laptops, classroom hubs, school edge servers, district-level aggregator.

**Challenges:** Sensitive student data, heterogeneous devices, different learning patterns, fairness, client dropout.

**Requirements:** Student performance or engagement prediction, personalized FL, privacy budget management, lightweight on-device inference.

### Option 4: Federated Home Energy Optimization

**Goal:** Create a federated forecasting system where smart meters, thermostats, home hubs, and neighborhood gateways collaboratively learn household energy patterns without exposing private home behavior.

**Participants:** Smart meters, thermostats, home energy hubs, neighborhood aggregators, utility server.

**Challenges:** Household privacy, time-series forecasting, device heterogeneity, bandwidth limits, asynchronous participation.

**Requirements:** Energy demand forecasting, secure aggregation, resource-aware training, compressed model for home edge deployment.

### Option 5: Your Own Idea

You may propose your own federated edge intelligence application if it has equivalent technical complexity and real-world relevance.

**Important:** Your own idea must include heterogeneous edge clients, a clear federated learning task, privacy or security concerns, hardware-aware optimization, and measurable evaluation metrics. Instructor approval is required during the proposal phase.

## Technical Requirements

### Core Components — All Required

#### 1. Federated Learning Framework

- Multi-device coordination protocol
- Client sampling or asynchronous participation
- Convergence monitoring and optimization
- Handling of non-IID client data

#### 2. Privacy Preservation System

- Differential privacy implementation or simulation
- Secure aggregation protocol or design
- Privacy budget management
- Threat model and privacy-risk discussion

#### 3. Hardware-aware Optimization

- Device capability profiling
- Resource-aware scheduling
- Dynamic model assignment
- Consideration of compute, memory, battery, and network limits

#### 4. Model Compression Pipeline

- Compression for different device classes
- Quantization, pruning, distillation, or another justified method
- Model size and latency analysis
- Accuracy-versus-efficiency tradeoff evaluation

#### 5. System Implementation

- Complete federated learning system code
- Simulation or real deployment environment
- Performance monitoring and analysis
- Reliability mechanisms for dropout, failures, or unstable clients

### Advanced Features — Choose 2–3

- Hierarchical Federated Learning
- Adaptive System Behavior
- Cross-platform Model Distillation
- Real-time System Monitoring
- Blockchain-based Participation Incentives

## Implementation Choices

### Real Hardware Deployment

- Deploy on actual edge devices, ideally at least 3 different device types.
- Demonstrate real-world constraints such as memory, compute, battery, or connectivity limits.
- **Additional credit may be awarded for successful real hardware deployment.**

### Comprehensive Simulation

- Simulate heterogeneous devices with different compute, memory, battery, and bandwidth profiles.
- Model realistic network behavior, including latency, dropouts, and partial participation.
- Provide clear evidence that the simulation captures meaningful edge-system constraints.

## Deliverables

### 1. Project Proposal — Due Week 8 — 3–4 pages

- Problem statement and motivation
- Selected project option and application scenario
- System architecture design
- Federated learning method and privacy approach
- Hardware-aware design plan
- Implementation plan and evaluation methodology

### 2. Final Implementation

- Complete federated learning system code
- Deployment scripts or simulation framework
- Privacy preservation mechanisms
- Model compression and optimization pipeline
- Performance monitoring tools
- Documentation and setup instructions
- System demonstration through screenshots, photos, videos, logs, dashboards, or simulation results

### 3. Final Report — Due December 15 — 8–10 pages

- System overview and architecture
- Federated learning design and implementation details
- Privacy, security, and threat model analysis
- Hardware-aware optimization and compression results
- Performance evaluation and comparison with baselines
- Implementation analysis, lessons learned, and future work

## Grading

| Component | Weight | Focus |
| --- | ---: | --- |
| System Architecture | 25% | Design quality, scalability, device heterogeneity, and real-world applicability. |
| Technical Implementation | 30% | Code quality, system functionality, federated learning implementation, and innovation. |
| Privacy & Security | 15% | Privacy preservation effectiveness, threat modeling, secure aggregation, and privacy budget analysis. |
| Performance Analysis | 15% | Evaluation quality, benchmarks, system metrics, compression analysis, and meaningful insights. |
| System Deployment | 10% | Practical deployment, realistic simulation, reproducibility, and system reliability. |
| Communication | 5% | Report clarity, presentation quality, documentation, and explanation of design choices. |

### Grade Expectations

**A — 90–100%:** Novel and well-engineered system with sophisticated implementation, strong privacy preservation, thoughtful hardware-aware design, thorough analysis, and professional documentation.

**B — 80–89%:** Well-designed system with solid implementation, effective privacy mechanisms, good performance analysis, and clear documentation.

**C — 70–79%:** Basic system meeting the core requirements, with functional implementation, basic privacy preservation, and adequate analysis.

**D/F — Below 70%:** Incomplete or flawed system with significant implementation issues, inadequate privacy preservation, weak evaluation, or poor documentation.

## Suggested Evaluation Metrics

| Category | Example Metrics |
| --- | --- |
| Model Quality | Accuracy, F1-score, AUC, MAE, RMSE, MAPE, task-specific recall. |
| Federated Learning | Convergence rounds, client participation rate, robustness to non-IID data, dropout tolerance. |
| System Performance | Communication cost, round time, inference latency, memory usage, energy or battery impact. |
| Privacy & Security | Privacy budget, accuracy impact of privacy mechanisms, threat analysis, poisoning or inference-attack simulation. |
| Compression | Model size reduction, latency reduction, bandwidth savings, accuracy loss after compression. |

## Collaboration Policy

- **Projects may be completed individually or in teams, according to instructor guidance.**
- External collaboration, including industry partners or other students, is permitted only with clear attribution.
- Open-source tools and datasets may be used, but they must be properly cited.
- Each team member must clearly explain their individual contribution in the final report.
