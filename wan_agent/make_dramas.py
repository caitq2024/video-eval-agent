#!/usr/bin/env python3
"""两段 H3 短剧（4 幕 × 15s 一体生成）的分镜元数据与 T2VA prompt。
每幕单镜长镜头（无内部切镜，与评估 span 切分兼容），幕间 cut 拼接。
prompt 遵循官方 h3-prompt-writing：三字段、[Shot 1] 开头、<d>[Chinese]对白、
画面文字双引号、运镜三维度句式、配乐无抽象情绪词。"""
import json
import os

from common import OUT_ROOT

D1_OWNER = ('a weary middle-aged noodle shop owner with short greying hair, '
            'a white apron over a dark T-shirt and rolled-up sleeves')
D1_WOMAN = ('a young woman with shoulder-length damp black hair, wearing a '
            'rain-soaked grey blazer over a white blouse')

D2_MAN = ('a young male office worker with tousled black hair, a wrinkled '
          'white shirt and a loosened navy tie')

DRAMAS = {
 'd1': {
  'idea': '短剧《深夜面馆》：雨夜将打烊的面馆，失业的年轻女子进店点一碗阳春面，'
          '老板默默多加两个荷包蛋，两人一段对话后，清晨雨停她带着暖意离开。',
  'title': 'Midnight Noodle Shop', 'title_zh': '深夜面馆',
  'style': 'live-action warm slice-of-life drama',
  'shots': [
   dict(shot_id=1, duration_s=15, camera='tracking', motion_level='low',
        transition_to_next='cut',
        expected_subjects=['a middle-aged noodle shop owner',
                           'a young woman in a grey blazer'],
        wan_prompt=('Late rainy night inside a tiny warm noodle shop about to '
                    'close. A weary middle-aged owner in a white apron wipes the '
                    'counter. A young woman in a rain-soaked grey blazer pushes '
                    'the glass door open, the entrance bell rings, she sits at '
                    'the counter and orders plain noodles. Warm tungsten light '
                    'against the blue rainy street.'),
        h3_prompt=(
'integrated_multimodal_description: [Shot 1] Live-action, cinematic slice-of-life '
'drama with warm tungsten interior light against a blue rainy night street, a '
'single continuous 15-second shot. A medium-wide composition inside a tiny '
'late-night noodle shop: ' + D1_OWNER + ' (S1) wipes the worn wooden counter '
'beside a steaming stockpot, stools empty, a paper sign on the wall reading '
'"深夜面馆" and a smaller one reading "阳春面 8元". Rain streaks down the glass '
'door. At around three seconds ' + D1_WOMAN + ' (S2) pushes the door open, the '
'entrance bell jingles, and she pauses at the doorway dripping. The camera '
'trucks right with small amplitude at slow speed, following her as she walks '
'to a counter stool, sets down her handbag and sits. She says quietly: '
'<d>[Chinese] 老板，一碗阳春面。</d> The owner (S1) nods and answers warmly: '
'<d>[Chinese] 好嘞，马上就好。</d> then turns to the stockpot, steam rising '
'around his face as he lifts the lid.\n\n'
'overall_soundscape: Steady rain patters on the awning and glass door, and the '
'entrance bell jingles once. A gas stove hisses beneath a bubbling stockpot '
'while wet shoes squeak softly on tile and a handbag thumps gently onto the '
'counter.\n\n'
'non_diegetic_music: A sparse solo piano with widely spaced warm chords at a '
'slow tempo, held at low volume beneath the rain.')),
   dict(shot_id=2, duration_s=15, camera='static', motion_level='low',
        transition_to_next='cut',
        expected_subjects=['a middle-aged noodle shop owner',
                           'a young woman in a grey blazer',
                           'a bowl of noodles'],
        wan_prompt=('Inside the noodle shop the owner cooks noodles in rolling '
                    'water, glances at the young woman who wipes her eyes while '
                    'staring at her phone. He drains the noodles, quietly adds '
                    'two poached eggs, carries the steaming bowl over and sets '
                    'it before her, telling her the eggs are on the house.'),
        h3_prompt=(
'integrated_multimodal_description: [Shot 1] Live-action, cinematic slice-of-life '
'drama with warm tungsten light and gentle film grain, a single continuous '
'15-second shot. A medium composition from behind the counter of a tiny '
'late-night noodle shop: ' + D1_OWNER + ' (S1) lowers a wire basket of noodles '
'into rolling water, steam swirling upward. In the mid-ground ' + D1_WOMAN +
' sits at the counter, head down, thumb scrolling her phone; she wipes the '
'corner of her eye with her sleeve. The owner glances at her, says nothing, '
'drains the noodles into a white bowl, ladles clear broth over them, then '
'quietly slides two poached eggs on top. The camera holds a static shot as he '
'walks the steaming bowl to her and sets it down with both hands, saying: '
'<d>[Chinese] 蛋是店里送的，趁热吃。</d> She looks up, surprised, and murmurs: '
'<d>[Chinese] 谢谢……</d> as steam drifts across her face.\n\n'
'overall_soundscape: Water boils in rolling bubbles and the wire basket clinks '
'against the pot rim, followed by broth being ladled into a ceramic bowl. Rain '
'continues faintly against the windows while the bowl lands on the wooden '
'counter with a soft thud.\n\n'
'non_diegetic_music: A slow solo piano joined by a single soft cello line, '
'sparse notes at low volume with a slight swell as the bowl is set down.')),
   dict(shot_id=3, duration_s=15, camera='slow_pan', motion_level='low',
        transition_to_next='cut',
        expected_subjects=['a middle-aged noodle shop owner',
                           'a young woman in a grey blazer',
                           'a bowl of noodles'],
        wan_prompt=('The young woman eats noodles and begins to sob, confessing '
                    'she lost her job today after five years in the city. The '
                    'owner sits down across the counter, pours her a cup of hot '
                    'tea and tells her he also lost a job ten years ago before '
                    'opening this shop, encouraging her to take it slow.'),
        h3_prompt=(
'integrated_multimodal_description: [Shot 1] Live-action, cinematic slice-of-life '
'drama with warm tungsten light and shallow depth of field, a single continuous '
'15-second shot. A medium close composition across the counter of the noodle '
'shop: ' + D1_WOMAN + ' (S2) lifts noodles with chopsticks, eats a mouthful, '
'then her shoulders tremble and she presses the back of her hand to her mouth. '
'She says through a tight throat: <d>[Chinese] 我今天被公司裁了……来这个城市'
'五年了。</d> ' + D1_OWNER + ' (S1) sets a cup of hot tea in front of her, '
'settles onto the stool across the counter, and replies in a low steady voice: '
'<d>[Chinese] 十年前我也丢过工作，后来才有了这家店。慢慢来，天亮了路就多了。'
'</d> She nods slowly, wiping her eyes, and picks the chopsticks back up. The '
'camera pans left with small amplitude at slow speed across the two of them as '
'steam rises between their faces.\n\n'
'overall_soundscape: Chopsticks click against the ceramic bowl and noodles are '
'drawn up with a soft slurp, followed by a quiet sob and a sniff. Hot tea pours '
'into a cup with a thin trickle while rain softens to scattered drips outside.\n\n'
'non_diegetic_music: A muted piano and low cello play long overlapping notes at '
'a very slow tempo, rising slightly in volume mid-way and settling back down.')),
   dict(shot_id=4, duration_s=15, camera='static', motion_level='medium',
        transition_to_next='cut',
        expected_subjects=['a middle-aged noodle shop owner',
                           'a young woman in a grey blazer'],
        wan_prompt=('At dawn the rain has stopped. The young woman stands at the '
                    'shop door, turns back and bows slightly to the owner, '
                    'promising to come again. The owner waves from behind the '
                    'counter and wishes her luck. She walks out into the soft '
                    'morning light on the wet street as he flips the door sign.'),
        h3_prompt=(
'integrated_multimodal_description: [Shot 1] Live-action, cinematic slice-of-life '
'drama, dawn light replacing the night, a single continuous 15-second shot. A '
'medium-wide composition from inside the noodle shop toward the glass door: '
'the rain has stopped and pale morning light spills across the wet street '
'outside. ' + D1_WOMAN + ' (S2), her hair now dry, stands at the open door, '
'turns back toward the counter, bows slightly and says with a small smile: '
'<d>[Chinese] 面很好吃，我还会再来的。</d> ' + D1_OWNER + ' (S1) raises a hand '
'from behind the counter and answers: <d>[Chinese] 祝你顺利，常来！</d> She '
'steps out onto the glistening street and walks into the morning light. The '
'owner walks to the door and flips the hanging sign that reads "营业中". The '
'camera holds a static shot through the doorway as her figure recedes.\n\n'
'overall_soundscape: The entrance bell jingles as the door swings, and light '
'footsteps recede on the wet pavement outside. Early birdsong and a distant '
'first bus pass by while the hanging sign taps once against the glass.\n\n'
'non_diegetic_music: A warm solo piano melody at a slow tempo joined by soft '
'strings, rising in volume for the final seconds and ending on a sustained '
'chord.')),
  ]},
 'd2': {
  'idea': '短剧《别打开的快递盒》：深夜加班的白领收到贴着"别打开"纸条的快递盒，'
          '盒子会动、灯光闪烁步步惊心，打开却是一只小猫和同事们的百日加班惊喜。',
  'title': 'Do Not Open', 'title_zh': '别打开的快递盒',
  'style': 'live-action suspense-to-heartwarming office drama',
  'shots': [
   dict(shot_id=1, duration_s=15, camera='zoom', motion_level='low',
        transition_to_next='cut',
        expected_subjects=['a young male office worker', 'a cardboard box'],
        wan_prompt=('Late at night in a dark open-plan office lit by one desk '
                    'lamp, a young office worker in a wrinkled white shirt '
                    'notices a cardboard courier box on his desk with a sticky '
                    'note reading do-not-open. He looks around the empty '
                    'cubicles, lifts the box and shakes it gently near his ear, '
                    'then sets it down and stares.'),
        h3_prompt=(
'integrated_multimodal_description: [Shot 1] Live-action, cinematic suspense '
'drama with cool blue-grey tones and one warm desk lamp, a single continuous '
'15-second shot. A medium composition in a dark open-plan office at night: '
+ D2_MAN + ' (S1) sits at his desk among rows of unlit cubicles, a monitor '
'glowing. A plain cardboard courier box sits on the desk with a yellow sticky '
'note reading "别打开". He notices it, frowns, stands halfway and scans the '
'empty office, then carefully lifts the box with both hands and shakes it '
'gently beside his ear. Something inside shifts with a soft slide. He mutters '
'to himself: <d>[Chinese] 谁放的这个……</d> and sets the box back down, staring '
'at the note. The camera zooms in with small amplitude at slow speed toward '
'the box and the note.\n\n'
'overall_soundscape: A low air-conditioning hum fills the empty office with a '
'faint monitor buzz. The cardboard box scrapes softly against the desk, '
'something inside slides with a muffled shift, and a distant elevator dings '
'once far away.\n\n'
'non_diegetic_music: A single sustained low synth drone with a slow sparse '
'sub-bass pulse, minimal and quiet throughout.')),
   dict(shot_id=2, duration_s=15, camera='static', motion_level='low',
        transition_to_next='cut',
        expected_subjects=['a young male office worker', 'a cardboard box'],
        wan_prompt=('The office worker has placed the cardboard box on a glass '
                    'meeting-room table and watches it from the doorway. The '
                    'box twitches slightly on its own, the ceiling light '
                    'flickers twice, and he stumbles back a step, grabbing a '
                    'long umbrella and holding it out like a sword toward the '
                    'box.'),
        h3_prompt=(
'integrated_multimodal_description: [Shot 1] Live-action, cinematic suspense '
'drama with cool tones and a flickering fluorescent ceiling light, a single '
'continuous 15-second shot. A medium-wide composition through the glass wall '
'of a small meeting room at night: the cardboard box with its yellow "别打开" '
'note sits alone in the center of a glass table. ' + D2_MAN + ' (S1) stands at '
'the doorway, half hidden behind the frame, watching. The box twitches once, '
'then rocks slightly on its own; the ceiling light flickers twice, throwing '
'the room into brief darkness. He gasps: <d>[Chinese] 它动了？！</d> stumbles '
'back a step, grabs a long black umbrella leaning by the door and points it '
'toward the box like a sword, edging forward with tiny steps. The camera holds '
'a static shot through the glass wall for the whole duration.\n\n'
'overall_soundscape: The fluorescent tube buzzes and clicks as it flickers, '
'while cardboard rocks against the glass table with light knocks. A sharp '
'intake of breath is followed by shoe soles squeaking on the office floor and '
'the metallic rattle of an umbrella being snatched up.\n\n'
'non_diegetic_music: A quickening string ostinato over an irregular low pulse, '
'rising in volume with two sharp staccato hits synced to the light flickers.')),
   dict(shot_id=3, duration_s=15, camera='zoom', motion_level='medium',
        transition_to_next='cut',
        expected_subjects=['a young male office worker', 'a cardboard box',
                           'an orange kitten'],
        wan_prompt=('The office worker uses the umbrella tip to flip open the '
                    'cardboard box lid, then freezes: a tiny orange kitten pokes '
                    'its head out, blinking at the light and mewing. He drops '
                    'the umbrella, exhales in relief and slumps into a chair, '
                    'then reaches in and lifts out the kitten together with a '
                    'folded paper note.'),
        h3_prompt=(
'integrated_multimodal_description: [Shot 1] Live-action, cinematic suspense '
'drama easing into warmth, cool tones with a warm pool of light over the '
'meeting-room table, a single continuous 15-second shot. A medium close '
'composition: ' + D2_MAN + ' (S1) extends a long black umbrella at arm\'s '
'length and flips the cardboard box lid open with its tip, flinching away. A '
'beat of stillness — then a tiny orange kitten pokes its head over the box '
'edge, ears back, blinking at the light, and lets out a thin mew. He drops the '
'umbrella with a clatter, exhales hugely and slumps into an office chair, '
'laughing once: <d>[Chinese] 吓死我了……就这？</d> He leans in, lifts the kitten '
'out with both hands, and notices a folded paper note inside the box, picking '
'it up with two fingers while cradling the kitten against his chest. The '
'camera zooms in with small amplitude at slow speed on the man and the kitten.\n\n'
'overall_soundscape: Cardboard flaps pop open and an umbrella clatters onto '
'the floor, followed by a thin kitten mew and a long relieved exhale. Office '
'chair wheels roll on the hard floor and paper crinkles as the note is picked '
'up.\n\n'
'non_diegetic_music: The string ostinato cuts off abruptly at the reveal, '
'replaced after a beat by light plucked pizzicato notes at a moderate tempo '
'and soft volume.')),
   dict(shot_id=4, duration_s=15, camera='slow_pan', motion_level='medium',
        transition_to_next='cut',
        expected_subjects=['a young male office worker', 'an orange kitten',
                           'colleagues'],
        wan_prompt=('The office worker reads the unfolded note aloud, holding '
                    'the orange kitten. The meeting room lights come on and a '
                    'group of colleagues bursts in from behind the door with '
                    'phone lights, applauding and celebrating his hundredth day '
                    'of overtime. He laughs and holds the kitten up as they '
                    'crowd around.'),
        h3_prompt=(
'integrated_multimodal_description: [Shot 1] Live-action, cinematic warm office '
'drama, lights coming up from cool to warm, a single continuous 15-second '
'shot. A medium composition in the meeting room: ' + D2_MAN + ' (S1) cradles '
'the tiny orange kitten in one arm and unfolds the paper note with his free '
'hand; the note shows handwritten characters reading "它叫勇气，带它回家吧". He '
'reads it aloud softly: <d>[Chinese] 它叫勇气，带它回家吧……</d> The ceiling '
'lights snap on, and a group of five colleagues in office wear bursts in from '
'behind the door with phone flashlights waving, applauding. A female colleague '
'(S2) calls out: <d>[Chinese] 加班一百天纪念日快乐！</d> He laughs, shakes his '
'head and answers: <d>[Chinese] 你们这群家伙……</d> lifting the kitten up as '
'they crowd around the table, someone patting his shoulder. The camera pans '
'right with small amplitude at slow speed across the celebrating group.\n\n'
'overall_soundscape: Paper unfolds with a crisp rustle and a kitten mews '
'against fabric. Ceiling lights click on with a hum, the door bangs open, and '
'overlapping applause, laughter and cheerful chatter fill the small room.\n\n'
'non_diegetic_music: A bright acoustic guitar strum pattern with hand claps at '
'a moderate tempo, swelling in volume as the group enters and ending on an '
'open ringing chord.')),
  ]},
}

for pid, d in DRAMAS.items():
    pdir = os.path.join(OUT_ROOT, pid)
    os.makedirs(pdir, exist_ok=True)
    mp = os.path.join(pdir, 'shots.json')
    if os.path.exists(mp):
        old = json.load(open(mp))
        d = {**old, **d}          # 保留既有 generation/films 记录
    json.dump(d, open(mp, 'w'), ensure_ascii=False, indent=1)
    print(pid, 'shots.json written,', len(d['shots']), 'acts')
print('done')
