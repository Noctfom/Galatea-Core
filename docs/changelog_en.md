# 📝 Changelog

> This document records the version update history of Galatea-Core.

---

## [v3.0] - 2026-05 (Major Release)

### ✨ Major Additions

- **📦 One-Click Package & Portable Environment**: Built-in Python 3.11 portable environment, zero configuration, download and extract to use
- **🧪 Complete Documentation System**: Full Chinese illustrated documentation (Quick Start, Feature Guide, Architecture, Special Handling, Changelog) + English README
- **🌐 Online Dynamic Environment Builder**: YGOProDeck API integration, supports batch auto-fetching decks by tournament/format type, with background daemon for scheduled auto-updates
- **🧠 Virtual Mix Pool Builder**: Cross-pool mixing recipe system, creates arbitrary-ratio deck mix pools, cross-pool battles without physical file movement
- **📦 Model Deployment & Packaging Module**: `.gkg` format model package export and selective import, including models, knowledge base, and staple pool in one-click share deployment

### 🔧 Feature Enhancements

- **WebUI Comprehensive Upgrade**:
  - 🗃️ Assets & Deck Management: New staple pool config, holographic deck viewer, dynamic weight scheduler, online fetch module
  - ⚔️ Launch & Monitor Hub: New process management (PID tracking/emergency abort/zombie purge), live terminal dual-column view (full logs + alert extraction)
  - 👁️ Holographic Replay: SVG board rendering refactored, supports ghost X-ray effects, action arrow connections, real-time coordinate mapping
  - 📉 Training Manifold: Embedded TensorBoard directly viewable
  - 📈 Meta Dashboard: Multi-source merged analysis, dual deck PK comparison
- **142 Announce Pool complete rewrite**: RPN reverse Polish expression virtual machine rewritten, added three-layer filtering (RPN filtering + common-sense candidate pool + weighted priority sorting), crash prevention fallback
- **Multi-select & 142 Weighted Priority Sorting**: Intelligent priority ranking by card source (own deck > known hand > public zones > staple pool), greatly improving announcement accuracy

### 🚀 Performance Optimizations

- **Data processing optimization**: Significantly optimized CPU usage for message parsing and feature encoding
- **Holographic replay rendering optimization**: SVG layer rendering refactored, large file loading speed significantly improved
- **WebUI responsiveness**: Cache strategy and log truncation logic improved, no lag during long-running operations
- **Memory management**: Multi-process worker memory reuse improvements

### 🔧 Under-the-Hood

- **Algorithm logic refinements**: PPO training stability improvements, reward function fine-tuning
- **Linux environment support**: Provided `ocgcore.so` (not fully tested), expanding cross-platform capability

---

## [v2.8] - 2026-04

### ✨ New Features
- **WebUI Console**: Brand new Streamlit Web interface with complete training and management functions
  - 📈 Meta Dashboard: Real-time win rate stats and counter matrix
  - 📉 Training Manifold: Embedded TensorBoard monitoring
  - ⚔️ Launch & Monitor Hub: One-click training/arena
  - 🔄 Resource Sync Hub: Auto-update card DB and scripts
  - 🧠 Semantic Knowledge Engine: Lua script semantic parsing
  - 📁 Storage & Logs: File management
- **142 Announce Handling**: Staple pool fallback mechanism to resolve card announcement effects

### 🔧 Optimizations
- Bilingual Chinese/English UI support
- Real-time terminal log mapping

---

## [v2.7] - 2026-04

### 🔧 Optimizations
- **Training reward stability optimization**: Improved reward function, reduced training fluctuation
- **Reduced game step limit**: Lowered from 3000 to 2000 steps, faster training speed

### 🐛 Fixes
- Fixed memory leak after prolonged training

---

## [v2.6] - 2026-04

### ✨ New Features
- **League training mechanism**: Mixed opponent training (Self-play 60% + Historical models 25% + RuleBot 15%)
- **Async inference server**: Centralized GPU inference, VRAM usage reduced by 70%+

### 🔧 Optimizations
- Complete training framework refactoring
- Self-play matching logic upgraded

---

## [v2.5] - 2026-03

### ✨ New Features
- **Hand tracker logic**: Track known cards in opponent's hand

### 🔧 Optimizations
- Field reading processing optimization
- Card position encoding improvements

---

## [v2.4] - 2026-03

### ✨ New Features
- **Auto-update tool**: `python main.py update` command support
- **Semantic effect parsing**: Improved Lua script parsing capability

### 🔧 Optimizations
- Effect semantic capability enhanced
- Knowledge base structure optimized

---

## [v2.3] - 2026-02

### 🐛 Fixes
- Fixed card reading errors
- Fixed field cognition issues

### 🔧 Optimizations
- Reward distribution mechanism optimization
- Feature encoding stability improvements

---

## [v2.2] - 2026-02

### ✨ New Features
- **Card selection effect learning**: SELECT_CARD type effect learning support
- **Chain cognition**: Improved chain stack understanding

### 🔧 Optimizations
- Action space encoding optimization

---

## [v2.1] - 2026-01

### ✨ New Features
- **Material selection learning**: DFS algorithm for Synchro/Link/Xyz material selection

### 🔧 Optimizations
- Memory management optimization
- Multi-process stability improvements

### 🐛 Fixes
- Fixed various training bugs

---

## [v2.0] - 2026-01

### ✨ New Features
- **Card effect reading & parsing**: Lua script semantic parsing system
- **Universal model architecture**: No longer deck-dependent, truly universal AI

### 🔧 Optimizations
- Neural network architecture refactored
- Feature encoding system redesigned

---

## [v1.x] - 2025

### Initial Version
- Basic PPO training framework
- OCGCore environment wrapper
- Simple RuleBot battles

---

## Version Naming Convention

- **Major version**: Significant architecture changes
- **Minor version**: New feature additions
- **Patch number**: Bug fixes and minor optimizations

---

## Next Steps

- 🚀 Return to [Quick Start](quickstart_en.md) to begin training
- 📚 Read [Feature Guide](features_en.md) for usage
- 🔧 Read [Architecture](architecture_en.md) for framework internals
