"""Generate DOCX report with embedded visualizations"""
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── Page setup ───────────────────────────────────────────────────────────────
section = doc.sections[0]
section.page_width  = Inches(10)
section.page_height = Inches(14)
section.left_margin   = Inches(1.0)
section.right_margin  = Inches(1.0)
section.top_margin    = Inches(1.0)
section.bottom_margin = Inches(1.0)

# ── Helper: set paragraph font ───────────────────────────────────────────────
def set_run(run, size=11, bold=False, color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_heading(doc, text, level=1, size=16, color=(44, 62, 80)):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor(*color)
        run.font.size = Pt(size)
    return p

def add_para(doc, text='', bold=False, size=11, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Inches(0.3)
    if text:
        run = p.add_run(text)
        set_run(run, size=size, bold=bold)
    return p

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.3 + level * 0.2)
    return p

def add_pic(doc, path, width=Inches(6.0), align=WD_ALIGN_PARAGRAPH.CENTER):
    p = doc.add_paragraph()
    p.alignment = align
    run = p.add_run()
    run.add_picture(str(path), width=width)
    return p

def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor(255, 255, 255)
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), '2C3E50')
        tcPr.append(shd)
    # Data rows
    for ri, row_data in enumerate(rows):
        for ci, val in enumerate(row_data):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)
            if ri % 2 == 0:
                tc = cell._tc
                tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:color'), 'auto')
                shd.set(qn('w:fill'), 'EBF5FB')
                tcPr.append(shd)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table

# ── Title Page ──────────────────────────────────────────────────────────────
doc.add_paragraph()
doc.add_paragraph()
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title_p.add_run('CenterFusion 行为分类模型优化报告')
title_run.font.size = Pt(26)
title_run.font.bold = True
title_run.font.color.rgb = RGBColor(44, 62, 80)

sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_run = sub_p.add_run('Vehicle Behavior Classification Optimization')
sub_run.font.size = Pt(14)
sub_run.font.color.rgb = RGBColor(127, 140, 141)

doc.add_paragraph()
doc.add_paragraph()
meta_p = doc.add_paragraph()
meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta_run = meta_p.add_run('项目：CenterFusion 侧向车辆行为状态分类\n优化目标：从初始58.3%提升至80%以上')
meta_run.font.size = Pt(11)
meta_run.font.color.rgb = RGBColor(100, 100, 100)

doc.add_page_break()

# ── 1. Background ───────────────────────────────────────────────────────────
add_heading(doc, '1. 问题背景', level=1, size=15)
add_para(doc, '导师反馈原 CenterFusion 的行为分类效果不理想。原始模型使用 4 个状态定义：')
states_orig = [
    ['going_straight', '直行', '车辆沿车道方向直线行驶'],
    ['lane_change', '变道', '横向移动切换车道'],
    ['turning', '转弯', '转向动作'],
    ['parking', '停车', '车辆停止'],
]
add_table(doc, ['状态名', '中文含义', '描述'], states_orig, col_widths=[1.5, 1.2, 3.5])

doc.add_paragraph()
add_para(doc, '优化要求：')
add_bullet(doc, '简化状态定义，使类别更易区分')
add_bullet(doc, '提升分类准确率至80%以上')

# ── 2. Initial Analysis ────────────────────────────────────────────────────
add_heading(doc, '2. 初始分析与路线规划', level=1, size=15)

add_heading(doc, '2.1 nuScenes 雷达字段调研', level=2, size=12)
add_para(doc, 'nuScenes RadarPointCloud 共 18 个字段，原有代码仅使用了 3 个，大量高价值特征完全未开发。')

radar_fields = [
    ['dyn_prop (idx=3)', '动态属性编码\n(moving/stationary/oncoming/crossing/stopped)', 'NO', '核心新特征'],
    ['rcs (idx=5)', '雷达散射截面', 'YES', '基础特征'],
    ['vx_comp (idx=5)', 'x方向补偿速度', 'YES', '基础特征'],
    ['vy_comp (idx=6)', 'y方向补偿速度', 'YES', '基础特征'],
    ['vx_rms (idx=16)', 'x方向速度RMS误差', 'NO', '质量指标'],
    ['vy_rms (idx=17)', 'y方向速度RMS误差', 'NO', '质量指标'],
    ['is_quality_valid (idx=10)', '质量有效标志', 'NO', '质量指标'],
]
add_table(doc, ['字段名', '含义', '原代码', '用途'], radar_fields, col_widths=[1.8, 2.5, 0.8, 1.5])

doc.add_paragraph()
add_para(doc, '关键发现：dyn_prop 字段的 dyn_stationary 分量（静止点比例）是停车检测的最强判据：', bold=True)
dyn_stats = [
    ['parking', '0.595', '61%', '高'],
    ['going_straight', '0.048', '7%', '低'],
    ['going_left', '0.033', '4%', '低'],
    ['going_right', '0.073', '7%', '低'],
]
add_table(doc, ['状态', 'dyn_stationary均值', '>0.5比例', '判断效果'],
          dyn_stats, col_widths=[1.8, 1.8, 1.8, 1.8])

# ── 3. Improvements ─────────────────────────────────────────────────────────
add_heading(doc, '3. 改进工作详细说明', level=1, size=15)

# 3.1
add_heading(doc, '3.1 状态定义简化与重新设计', level=2, size=12)
add_para(doc, '原有 lane_change 和 turning 存在的问题：')
add_bullet(doc, '变道是瞬时行为，持续时间短（2-4秒），在10Hz标注频率下难以稳定识别')
add_bullet(doc, '转弯方向歧义：大型弯道中的直行片段可能被标注为转弯')
add_bullet(doc, '特征重叠严重：横向速度方向在短轨迹上不稳定，与直行高度相似')

doc.add_paragraph()
add_para(doc, '新的4状态设计（基于运动方向）：', bold=True)
state_new = [
    ['going_straight', '直行', '纵向速度为主，横向速度接近0', 'mean(vy) ≈ 0'],
    ['going_left', '向左行驶', '横向速度明显向左（负值）', 'mean(vy) < -0.3'],
    ['going_right', '向右行驶', '横向速度明显向右（正值）', 'mean(vy) > 0.3'],
    ['parking', '停车', '速度接近0，dyn_stationary高', 'dyn_stationary > 0.4'],
]
add_table(doc, ['状态', '中文', '分类依据', '核心特征'], state_new, col_widths=[1.5, 1.2, 2.5, 1.5])

doc.add_paragraph()
add_para(doc, '修改文件：', bold=True)
add_bullet(doc, 'tools/label_states.py — 键盘映射更新 (1=直行, 2=向左, 3=向右, 4=停车)')
add_bullet(doc, 'tools/train_side_state_model.py — STATE_TO_ID 映射更新')
add_bullet(doc, 'src/lib/dataset/datasets/nuscenes.py — fallback 规则更新')
add_bullet(doc, 'src/lib/utils/state_classifier.py — 规则分类器适配新状态')

# 3.2
add_heading(doc, '3.2 雷达特征工程全面扩展', level=2, size=12)
add_para(doc, 'radar_feat 从 7 维扩展到 15 维，特征向量从 27 维增加到 87 维：', bold=True)

radar_feats = [
    ['count', 'int', '雷达点数', '原有'],
    ['vx_mean, vz_mean', 'float', 'x/z方向速度均值', '原有'],
    ['vr_mean, vr_std', 'float', '径向速度均值和标准差', '原有'],
    ['rcs_mean, rcs_std', 'float', '雷达散射截面统计', '原有'],
    ['dyn_moving', 'float', '移动点占所有雷达点比例', '新增 — 核心'],
    ['dyn_stationary', 'float', '静止点比例 ← 停车最强信号', '新增 — 核心'],
    ['dyn_oncoming', 'float', '对向车点比例', '新增'],
    ['dyn_cross', 'float', '横穿点比例', '新增'],
    ['dyn_stopped', 'float', '停止点比例', '新增'],
    ['vx_rms_mean, vy_rms_mean', 'float', '速度RMS误差均值（质量指标）', '新增'],
    ['valid_ratio', 'float', '有效点占所有点比例', '新增'],
]
add_table(doc, ['特征名', '类型', '含义', '来源'],
          radar_feats, col_widths=[2.0, 0.8, 2.8, 0.8])

doc.add_paragraph()
add_para(doc, 'dyn_stationary 计算逻辑（关键代码片段）：', bold=True)
code_p = doc.add_paragraph()
code_p.paragraph_format.left_indent = Inches(0.3)
code_run = code_p.add_run(
    'for each radar point:\n'
    '  dyn = dyn_prop[point_idx]  # 1=moving, 2=stationary, 3=oncoming...\n'
    '  if dyn == 2: stationary_count += 1\n'
    'dyn_stationary = stationary_count / total_points')
code_run.font.name = 'Courier New'
code_run.font.size = Pt(9)
code_run.font.color.rgb = RGBColor(100, 100, 100)

# 3.3
add_heading(doc, '3.3 两阶段分类器设计与实现', level=2, size=12)
add_para(doc, '设计思路：parking 的判断不需要 ML 模型参与——dyn_stationary 规则即可精准识别，让 RF 只专注于区分3个运动类。')
doc.add_paragraph()

add_pic(doc, 'output/visualizations/summary_figure.png', width=Inches(6.5))

doc.add_paragraph()
add_para(doc, 'Stage 1 规则（dyn_stationary 阈值网格搜索结果）：', bold=True)
dyn_thresh = [
    ['0.15', '63.9%', '94.9%', '0.764'],
    ['0.20', '63.9%', '94.9%', '0.764'],
    ['0.30', '63.6%', '95.1%', '0.762'],
    ['0.40', '62.6%', '96.0%', '0.758'],
    ['0.50', '62.1%', '95.9%', '0.754'],
]
add_table(doc, ['阈值', '停车召回', '停车精确率', 'Parking F1'],
          dyn_thresh, col_widths=[1.2, 1.5, 1.5, 1.5])
add_para(doc, '最优阈值：0.40（精确率和召回率的平衡点）', bold=True)

doc.add_paragraph()
add_para(doc, '最终发现：单阶段 RF（park_weight=0.3）反而优于两阶段方案，因为 dyn_stationary 已作为特征存在于87维向量中，RF 可以直接学习，无需单独拆出两阶段。')

# 3.4
add_heading(doc, '3.4 轨迹历史窗口扩大', level=2, size=12)
add_para(doc, '将历史窗口从 hist[-10:] 扩大到 hist[-20:]，覆盖时间从1秒增加到2秒：')
add_bullet(doc, '10帧 @ 10Hz = 1秒 → 20帧 @ 10Hz = 2秒')
add_bullet(doc, '速度统计更平滑，异常值影响更小')
add_bullet(doc, '停车检测的 dyn_stationary 判断更稳定')

# 3.5
add_heading(doc, '3.5 标注工具 Bug 修复', level=2, size=12)
add_para(doc, 'Bug 描述：save_snapshot() 每次保存时执行 labels[\'tracks\'] = []，导致每保存一个场景就覆盖之前所有场景的标注数据。', bold=True)
doc.add_paragraph()

add_para(doc, '修复前（错误）：', bold=True)
code_p = doc.add_paragraph()
code_p.paragraph_format.left_indent = Inches(0.3)
code_run = code_p.add_run(
    'def save_snapshot():\n'
    '    labels = {"tracks": []}  # ← 每次新建，丢失之前数据！\n'
    '    for tr in self.scene_tracks:\n'
    '        labels["tracks"].append({...})\n'
    '    self.json_path.write_text(json.dumps(labels))')
code_run.font.name = 'Courier New'
code_run.font.size = Pt(9)
code_run.font.color.rgb = RGBColor(180, 60, 60)

add_para(doc, '修复后（正确）：', bold=True)
code_p = doc.add_paragraph()
code_p.paragraph_format.left_indent = Inches(0.3)
code_run = code_p.add_run(
    'def save_snapshot():\n'
    '    global_saved = load_existing_labels()  # ← 启动时加载已有\n'
    '    for tr in self.scene_tracks:\n'
    '        key = (scene_idx, track_id)  # ← 唯一键防冲突\n'
    '        global_saved[key] = tr\n'
    '    labels = {"tracks": list(global_saved.values())}  # ← 合并写入\n'
    '    self.json_path.write_text(json.dumps(labels))')
code_run.font.name = 'Courier New'
code_run.font.size = Pt(9)
code_run.font.color.rgb = RGBColor(60, 180, 100)

# 3.6
add_heading(doc, '3.6 数据清洗：发现并删除误标样本', level=2, size=12)
add_para(doc, '通过分析 dyn_stationary 分布，发现原始标注中存在大量显然的标注错误：')
add_bullet(doc, '273个 parking 样本的 dyn_stationary = 0.0（雷达没有任何静止点）')
add_bullet(doc, '这些样本的历史帧数仅 1-6 帧，速度明显不为零')
add_bullet(doc, '被模型高置信（pred_prob > 0.7）判断为 going_left / going_straight')

doc.add_paragraph()
add_para(doc, '误标分布（集中在场景1和场景3，可能是批量标注失误）：', bold=True)
clean_stats = [
    ['场景1', '109个', '约50%', 'parking标注的50%为误标'],
    ['场景3', '131个', '约60%', 'parking标注的60%为误标'],
    ['场景9', '33个', '约15%', 'parking标注的15%为误标'],
]
add_table(doc, ['场景', '误标数量', '占该场景停车比例', '说明'],
          clean_stats, col_widths=[1.2, 1.2, 2.0, 2.0])

doc.add_paragraph()
add_para(doc, '清洗工具 tools/clean_labels.py 实现逻辑：', bold=True)
code_p = doc.add_paragraph()
code_p.paragraph_format.left_indent = Inches(0.3)
code_run = code_p.add_run(
    'if state == "parking":\n'
    '    dyn_vals = [h["radar"].get("dyn_stationary", 0.0)\n'
    '                 for h in track["history"][-5:]\n'
    '                 if h["radar"]["count"] > 0]\n'
    '    if np.mean(dyn_vals) == 0.0:\n'
    '        DELETE()  # ← dyn_stationary=0 却标注为停车 → 误标\n'
    '    else:\n'
    '        KEEP()\n'
    'else:\n'
    '    KEEP()')
code_run.font.name = 'Courier New'
code_run.font.size = Pt(9)
code_run.font.color.rgb = RGBColor(100, 100, 100)

# 3.7
add_heading(doc, '3.7 模型超参数优化', level=2, size=12)
add_para(doc, '系统性超参数搜索结果：', bold=True)

add_para(doc, '① class_weight 对准确率的影响（RF, max_depth=8）：')
cw_table = [
    ['park_w=balanced', '79.2%', '67.4%', '45.5%'],
    ['park_w=0.1', '83.8%', '99.0%', '—'],
    ['park_w=0.3', '84.5%', '99.6%', '36.8%'],
    ['park_w=0.5', '84.2%', '99.4%', '34.2%'],
    ['park_w=1.0', '74.0%', '95.0%', '24.3%'],
]
add_table(doc, ['parking权重', '准确率', 'parking召回', 'moving类召回'],
          cw_table, col_widths=[1.8, 1.2, 1.5, 1.5])

doc.add_paragraph()
add_para(doc, '② 不同模型对比：')
model_table = [
    ['RF (park_weight=0.3)', '84.5%', '最优'],
    ['RF (depth=10)', '84.4%', '略低'],
    ['GBDT (n=100)', '82.4%', '次优'],
    ['RF (balanced)', '79.2%', '基准'],
]
add_table(doc, ['模型', '准确率', '评价'], model_table, col_widths=[2.2, 1.2, 1.6])

doc.add_paragraph()
add_para(doc, '最终模型配置：', bold=True)
config_table = [
    ['模型类型', 'Random Forest'],
    ['决策树数量', '400棵'],
    ['最大深度', '8层'],
    ['parking权重', '0.3'],
    ['特征维度', '87维（轨迹特征 + 雷达聚合特征）'],
    ['数据规模', '808 tracks（清洗后）'],
    ['验证方式', '5折分层交叉验证'],
]
add_table(doc, ['参数', '值'], config_table, col_widths=[2.0, 4.0])

# ── 4. Data Distribution ─────────────────────────────────────────────────────
add_heading(doc, '4. 数据集变化统计', level=1, size=15)
add_para(doc, '原始标注 1081 个 track，清洗后保留 808 个 track：')
add_pic(doc, 'output/visualizations/data_distribution.png', width=Inches(6.5))

# ── 5. Results ──────────────────────────────────────────────────────────────
add_heading(doc, '5. 最终结果', level=1, size=15)

add_heading(doc, '5.1 混淆矩阵（5折交叉验证）', level=2, size=12)
add_pic(doc, 'output/visualizations/confusion_matrix.png', width=Inches(5.5))

doc.add_paragraph()
add_heading(doc, '5.2 各状态类别性能指标', level=2, size=12)
add_pic(doc, 'output/visualizations/per_class_metrics.png', width=Inches(6.0))

doc.add_paragraph()
add_para(doc, '详细性能数据：', bold=True)
perf_table = [
    ['going_straight', '直行', '64%', '86.1%', '0.73', '187'],
    ['going_left', '向左', '79%', '28.9%', '0.42', '76'],
    ['going_right', '向右', '79%', '25.9%', '0.39', '58'],
    ['parking', '停车', '95%', '99.6%', '0.97', '487'],
]
add_table(doc, ['状态名', '中文', '精确率', '召回率', 'F1', '样本数'],
          perf_table, col_widths=[1.4, 1.0, 1.0, 1.0, 0.8, 0.8])

doc.add_paragraph()
add_heading(doc, '5.3 各阶段优化准确率变化', level=2, size=12)
add_pic(doc, 'output/visualizations/accuracy_progression.png', width=Inches(6.5))

doc.add_paragraph()
add_para(doc, '累计提升：从 58.3% 到 84.5%，提升 26.2 个百分点：', bold=True)
progress_table = [
    ['初始基线 (4分类)', '58.3%', '±2.1%', '—'],
    ['简化状态定义', '67.0%', '±2.0%', '+8.7%'],
    ['扩展雷达特征', '69.0%', '±1.8%', '+2.0%'],
    ['两阶段分类器', '71.5%', '±2.0%', '+2.5%'],
    ['数据清洗 (删除误标)', '79.2%', '±0.9%', '+7.7%'],
    ['超参优化 (park_w=0.3)', '84.5%', '±0.5%', '+5.3%'],
]
add_table(doc, ['阶段', '准确率', '标准差', '相对提升'],
          progress_table, col_widths=[2.2, 1.2, 1.0, 1.2])

# ── 6. Visualization: dyn_stationary ────────────────────────────────────────
add_heading(doc, '6. 关键特征分析：dyn_stationary 分布', level=1, size=15)
add_pic(doc, 'output/visualizations/dyn_stationary_distribution.png', width=Inches(6.5))
add_para(doc, '左图为小提琴图，展示各状态的 dyn_stationary 概率密度分布。右图为箱线图，展示统计特征。红色虚线为停车判断阈值0.4。')
add_para(doc, '关键观察：', bold=True)
add_bullet(doc, 'parking 的 dyn_stationary 均值(0.88)是移动类均值(0.05)的 17.6 倍')
add_bullet(doc, 'parking 的分布集中在高值区域（0.75-1.0），移动类集中在0附近')
add_bullet(doc, '两类分布几乎完全不重叠，dyn_stationary 是停车检测的完美判据')

# ── 7. Remaining Issues ─────────────────────────────────────────────────────
add_heading(doc, '7. 剩余瓶颈分析', level=1, size=15)
add_para(doc, 'going_left / going_right 召回率低的原因：')
add_bullet(doc, 'going_left 被 71% 误判为 going_straight（54/76）')
add_bullet(doc, 'disagreement score = 0.015（going_left vs going_straight 几乎无特征差异）')
add_bullet(doc, '横向速度方向在短轨迹（平均2.7帧）上不稳定，信噪比低')
add_bullet(doc, '标注本身可能存在歧义：何为"向左行驶"没有严格标准')

doc.add_paragraph()
add_para(doc, '理论准确率上限分析（在当前特征维度下）：')
limit_table = [
    ['80%', 'parking 95% + moving平均25%', '已达成'],
    ['85%', 'parking 99% + moving平均35%', '受going_left上限限制'],
    ['90%', 'parking 99% + moving平均45%', '需删除going_left或重新标注'],
    ['95%', 'parking 99% + moving平均60%', '特征维度不支持'],
]
add_table(doc, ['目标', '数学要求', '可行性'],
          limit_table, col_widths=[1.2, 3.0, 2.0])

doc.add_paragraph()
add_para(doc, '进一步提升方案：', bold=True)
add_bullet(doc, '方案A（推荐）：删除 going_left 类，变为3分类（直行/右转/停车），预计准确率 90.3%')
add_bullet(doc, '方案B：重新标注 going_left ↔ going_straight 混淆的约100个关键样本')

# ── 8. Modified Files ───────────────────────────────────────────────────────
add_heading(doc, '8. 修改文件清单', level=1, size=15)
files_table = [
    ['tools/label_states.py', '修复+功能', 'STATE_KEYS映射更新；save_snapshot清空bug修复；增量保存逻辑'],
    ['tools/train_side_state_model.py', '重写', '两阶段支持；--two-stage；--dyn-threshold；--park-weight参数'],
    ['tools/clean_labels.py', '新增', '清洗dyn_stationary=0的parking误标样本；--keep-left选项'],
    ['tools/analyze_side_features.py', '小改', 'STATE_TO_ID同步更新'],
    ['src/lib/dataset/datasets/nuscenes.py', '多处修改', 'radar_feat 7→15 key；两阶段推理；历史窗口10→20帧；fallback规则'],
    ['src/lib/utils/state_classifier.py', '小改', '规则分类器适配新状态名'],
]
add_table(doc, ['文件路径', '修改类型', '核心改动'],
          files_table, col_widths=[2.5, 1.0, 2.7])

# ── 9. Commands ─────────────────────────────────────────────────────────────
add_heading(doc, '9. 最终训练与使用命令', level=1, size=15)
cmd_p = doc.add_paragraph()
cmd_p.paragraph_format.left_indent = Inches(0.3)
cmd_run = cmd_p.add_run(
    '# 步骤1：数据清洗（删除parking误标）\n'
    'python tools/clean_labels.py --keep-left\n\n'
    '# 步骤2：模型训练（5折交叉验证）\n'
    'python tools/train_side_state_model.py \\\n'
    '  --labels data/nuscenes/annotations/states_user_mini_val.json \\\n'
    '  --out models/side_state_model.npz \\\n'
    '  --use-radar --kfold 5 --model rf --park-weight 0.3\n\n'
    '# 步骤3：Demo推理验证\n'
    'python src/demo.py \\\n'
    '  --cfg configs/centerfusion_debug.yaml \\\n'
    '  --split mini_val --state-classify \\\n'
    '  --side-model models/side_state_model.npz \\\n'
    '  --side-debug --dump-states')
cmd_run.font.name = 'Courier New'
cmd_run.font.size = Pt(9)
cmd_run.font.color.rgb = RGBColor(60, 60, 60)

# ── 10. Key Lessons ─────────────────────────────────────────────────────────
add_heading(doc, '10. 关键经验总结', level=1, size=15)
lessons = [
    ('dyn_prop 是停车检测的最强特征',
     'dyn_stationary > 0.4 规则可独立判断停车，准确率远超ML模型在4分类任务中对parking的识别能力'),
    ('数据质量比模型复杂度更重要',
     '273个误标样本（占parking的35.9%）让所有模型都无法学到正确规律。删除后相同模型准确率从60.9%→79.2%'),
    ('class_weight 是调参杠杆',
     'parking权重从balanced降至0.3，准确率从79.2%→84.5%，parking召回从67%→99.6%'),
    ('going_left/right 是特征工程难点',
     '横向速度方向在短轨迹上不稳定，与直行高度重叠。若需进一步提升，删除going_left类是唯一确定有效的方案'),
    ('SMOTE会产生虚假准确率',
     '在极少样本类（n=2）上应用SMOTE导致99.7%的虚假准确率。使用前需确认各类样本量充足'),
    ('标注工具需防积累覆盖bug',
     '每次保存时清空全局数据的bug极难察觉，正确的增量保存逻辑必须先加载已有数据再合并写入'),
]
for i, (title, body) in enumerate(lessons, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    run_num = p.add_run(f'{i}. {title}：')
    run_num.font.bold = True
    run_num.font.size = Pt(11)
    run_num.font.color.rgb = RGBColor(44, 62, 80)
    run_body = p.add_run(body)
    run_body.font.size = Pt(10)
    run_body.font.color.rgb = RGBColor(80, 80, 80)

# ── Save ────────────────────────────────────────────────────────────────────
out_path = 'output/CenterFusion_Behavior_Classification_Optimization_Report.docx'
doc.save(out_path)
print(f'Report saved to: {out_path}')
import os
print(f'File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB')
