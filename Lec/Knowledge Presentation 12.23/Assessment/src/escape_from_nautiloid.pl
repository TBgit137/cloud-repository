/* Escape from Nautiloid, by Luo Xiaoyu, Chen Zhuoxin, Luo Juntong */

/*
   入口谓词：start/0
   目前负责展示开场白与职业选择界面的文字，
   并在之后根据玩家选择初始化角色状态。
*/

:- dynamic player_character/1.

/* ----------------------------------------------------------------------
   角色与职业建模

   我们用一个结构体式的 term 来表示角色：

     character(Class, HP, AC, Weapons, Inventory, Skills, Action, BonusAction)

   - Class        : 职业标识（fighter | wizard | ranger）
   - HP           : 当前生命值
   - AC           : 防御值
   - Weapons      : 武器列表
   - Inventory    : 背包物品列表
   - Skills       : 技能列表
   - Action       : 动作点（0 或 1）
   - BonusAction  : 附赠动作点（0 或 1）
---------------------------------------------------------------------- */

% 通用“父类”：给定参数，构造一个角色，Action 与 BonusAction 初始为 1
make_character(Class, HP, AC, Weapons, Inventory, Skills,
               character(Class, HP, AC, Weapons, Inventory, Skills, 1, 1)).

/* ------------------------ 三个具体职业 ------------------------ */

% Fighter：HP 28，AC 10，巨剑，三个药水，四个技能
fighter(Character) :-
    Weapons   = ['Greatsword'],
    Inventory = ['Potion of Healing', 'Potion of Healing', 'Potion of Damage'],
    Skills    = ['Main Hand Attack', 'Lacerate', 'Pommel Strike', 'Second Wind'],
    make_character(fighter, 28, 10, Weapons, Inventory, Skills, Character).

% Wizard：HP 18，AC 5，法杖，两个治疗药水，六个法术/技能
wizard(Character) :-
    Weapons   = ['Staff'],
    Inventory = ['Potion of Healing', 'Potion of Healing'],
    Skills    = ['Fire Bolt', 'Ray of Frost', 'Thunderwave',
                 'Magic Missile', 'Shield', 'Misty Step'],
    make_character(wizard, 18, 5, Weapons, Inventory, Skills, Character).

% Ranger：HP 21，AC 8，匕首 + 短弓，两个治疗药水，五个技能
ranger(Character) :-
    Weapons   = ['Dagger', 'Shortbow'],
    Inventory = ['Potion of Healing', 'Potion of Healing'],
    Skills    = ['Hide', 'Sneak Attack (Melee)', 'Sneak Attack (Ranged)',
                 'Main Hand Attack', 'Shoot'],
    make_character(ranger, 21, 8, Weapons, Inventory, Skills, Character).

/* 根据玩家输入的数字创建并记录当前角色（后续可在交互逻辑中调用） */

set_player_class(1) :-
    fighter(Char),
    retractall(player_character(_)),
    assertz(player_character(Char)).

set_player_class(2) :-
    wizard(Char),
    retractall(player_character(_)),
    assertz(player_character(Char)).

set_player_class(3) :-
    ranger(Char),
    retractall(player_character(_)),
    assertz(player_character(Char)).

/* 获取当前玩家角色（如果未设置则失败） */
current_player(Char) :-
    player_character(Char).

/* 游戏入口：目前先只展示剧情与职业选择界面 */

start :-
    show_intro,
    nl, nl,
    show_class_selection.

/* 通用打字机式输出 */

typing_delay(0.01).

typewriter_print(Text) :-
    typing_delay(Delay),
    typewriter_print(Text, Delay).

typewriter_print(Text, Delay) :-
    string_chars(Text, Chars),
    maplist(typewriter_char(Delay), Chars).

typewriter_char(Delay, Char) :-
    put_char(Char),
    flush_output,
    sleep(Delay).

typewriter_line(Text) :-
    typewriter_print(Text),
    nl.

/* 打印开场白 */

show_intro :-
    typewriter_line('The familiar world vanished in a flash of thunder and alien light.'),
    typewriter_line('You wake to the cold, pulsating horror of the Nautiloid, a ship forged from flesh and metal.'),
    nl,
    typewriter_line('A searing phantom pain lingers behind your eye --a Illithid Tadpole now resides in your skull,'),
    typewriter_line('the Mind Flayers'' gruesome gift. You are a captive, a ticking clock, a meal.'),
    nl,
    typewriter_line('But fate has intervened. Shouts and tremors rock the vessel, its organic machinery failing.'),
    typewriter_line('The Mind Flayer Pod that held you is broken.'),
    nl,
    typewriter_line('The cage is open. Your memory is fractured, but your will to survive is whole.'),
    nl,
    typewriter_line('Who were you, before this nightmare began?').

/* 打印职业选择界面（仅文字展示） */

show_class_selection :-
    typewriter_line('Choose your class:'),
    nl,
    typewriter_line('1. Fighter'),
    typewriter_line('   - High hit points and heavy armor'),
    typewriter_line('   - Starts with a greatsword'),
    typewriter_line('   - Powerful muscles grant extraordinary jump distance'),
    typewriter_line('   - Excels at close-quarters combat'),
    nl,
    typewriter_line('2. Wizard'),
    typewriter_line('   - Low hit points and light armor'),
    typewriter_line('   - Starts with a wooden staff'),
    typewriter_line('   - Exceptional intellect allows spellcasting'),
    typewriter_line('   - Some spells consume spell slots'),
    nl,
    typewriter_line('3. Ranger'),
    typewriter_line('   - Moderate hit points and armor'),
    typewriter_line('   - Starts with a shortbow and dagger'),
    typewriter_line('   - Agile body favors light weapons and ambushes'),
    typewriter_line('   - Strikes from the shadows before foes even notice'),
    nl,
    typewriter_line('(Please choose your class by entering the corresponding number:)'),
    nl,
    get_class_choice.

/* ----------------------------------------------------------------------
   读取并验证玩家职业选择
---------------------------------------------------------------------- */

% 不断读取玩家输入，直到输入 1 / 2 / 3 为止
get_class_choice :-
    read(Choice),
    handle_class_choice(Choice).

handle_class_choice(1) :-
    set_player_class(1),
    current_player(Char),
    show_player_info(Char).

handle_class_choice(2) :-
    set_player_class(2),
    current_player(Char),
    show_player_info(Char).

handle_class_choice(3) :-
    set_player_class(3),
    current_player(Char),
    show_player_info(Char).

handle_class_choice(_) :-
    nl,
    typewriter_line('Invalid input. Please enter 1, 2, or 3.'),
    nl,
    get_class_choice.

/* 打印当前玩家的完整职业信息 */

show_player_info(character(Class, HP, AC, Weapons, Inventory, Skills, Action, BonusAction)) :-
    nl,
    typewriter_line('You have created your character:'),
    format(string(S1), 'Class: ~w', [Class]),
    typewriter_line(S1),
    format(string(S2), 'HP: ~w', [HP]),
    typewriter_line(S2),
    format(string(S3), 'AC: ~w', [AC]),
    typewriter_line(S3),
    format(string(S4), 'Weapons: ~w', [Weapons]),
    typewriter_line(S4),
    format(string(S5), 'Inventory: ~w', [Inventory]),
    typewriter_line(S5),
    format(string(S6), 'Skills: ~w', [Skills]),
    typewriter_line(S6),
    format(string(S7), 'Action: ~w', [Action]),
    typewriter_line(S7),
    format(string(S8), 'Bonus Action: ~w', [BonusAction]),
    typewriter_line(S8),
    nl,
    ask_confirm_class.

/* 询问并确认是否选择该职业 */

ask_confirm_class :-
    typewriter_line('Do you want to keep this class? (y/n)'),
    read(Ans),
    handle_confirm_answer(Ans).

handle_confirm_answer(y) :-
    nl,
    typewriter_line('Class confirmed.'),
    nl.

handle_confirm_answer(n) :-
    % 放弃当前职业，重新开始选择
    retractall(player_character(_)),
    nl,
    typewriter_line('Let\'s choose again.'),
    nl,
    show_class_selection.

handle_confirm_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_confirm_class.


