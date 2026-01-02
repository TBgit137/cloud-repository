:- encoding(utf8).
/* Escape from Nautiloid, by Luo Xiaoyu, Chen Zhuoxin, Luo Juntong */

/* Import PDDL planner interface */
:- use_module(pyperplan_runner).

/* ======================================================================
   Dynamic Predicate Declarations and Initialization
   ====================================================================== */

:- dynamic player_character/1.
:- dynamic cranial_valve_state/1.
:- dynamic combat_round/1.
:- dynamic skill_usage_count/2.
:- dynamic has_slate/1.
:- dynamic cleric_rescued/1.
:- dynamic current_combat/1.  % Track current combat type: imp_encounter | boss_encounter
:- dynamic enemy_instance/2.  % enemy_instance(Id, CharacterTerm).
:- dynamic skill/10.

:- discontiguous handle_player_choice/1.
:- discontiguous handle_insert_slate_answer/1.
:- discontiguous enemy_display_name/2.

/* Initialize door state to closed */
:- initialization(init_cranial_valve).

init_cranial_valve :-
    ( cranial_valve_state(_)
    -> true
    ;  assertz(cranial_valve_state(closed))
    ).

/* Initialize slate state */
:- initialization(init_slate_state).

init_slate_state :-
    ( has_slate(_)
    -> true
    ;  assertz(has_slate(false))
    ).

/* Initialize cleric rescue state */
:- initialization(init_cleric_state).

init_cleric_state :-
    ( cleric_rescued(_)
    -> true
    ;  assertz(cleric_rescued(false))
    ).

/* ======================================================================
   Character and Class Modeling

   We use a structured term to represent a character:

     character(Class, HP, AC, Weapons, Inventory, Skills, Action, BonusAction, SpellSlots, StatusEffects)

   - Class        : Class identifier (fighter | wizard | ranger | paladin | imp | cambion)
   - HP           : Current hit points
   - AC           : Armor class (defense value)
   - Weapons      : List of weapons
   - Inventory    : List of items in backpack
   - Skills       : List of skill IDs (e.g., [greatsword_attack, pommel_strike])
   - Action       : Action points (0 or 1)
   - BonusAction  : Bonus action points (0 or 1)
   - SpellSlots   : Current available spell slots (integer)
   - StatusEffects: List of current status effects (e.g., [bleed, ac_bonus(5), action_zero])
   ====================================================================== */

% Generic "parent class": given parameters, construct a character
% Action and BonusAction are initialized to 1, SpellSlots and StatusEffects must be specified externally
make_character(Class, HP, AC, Weapons, Inventory, Skills, SpellSlots,
               character(Class, HP, AC, Weapons, Inventory, Skills, 1, 1, SpellSlots, [])).

/* ------------------------ Four Playable Classes ------------------------ */

% Fighter: HP 40, AC 2, Greatsword, three potions, four skills, 0 spell slots
% Characteristics: High HP, high damage, frontline melee warrior
fighter(Character) :-
    Weapons   = ['Greatsword'],
    Inventory = ['Potion of Healing', 'Potion of Healing', 'Potion of Damage'],
    Skills    = ['Greatsword Attack', 'Pommel Strike', 'Lacerate', 'Second Wind'],
    make_character(fighter, 40, 2, Weapons, Inventory, Skills, 0, Character).

% Wizard: HP 25, AC 0, Staff, two healing potions, four spells/skills, 3 spell slots
% Characteristics: Glass cannon mage, relies on powerful spells and Shield to survive
wizard(Character) :-
    Weapons   = ['Staff'],
    Inventory = ['Potion of Healing', 'Potion of Healing'],
    Skills    = ['Fire Bolt', 'Witch Bolt', 'Thunderwave', 'Shield'],
    make_character(wizard, 25, 0, Weapons, Inventory, Skills, 3, Character).

% Ranger: HP 32, AC 1, Dagger + Shortbow, two healing potions, four skills, 0 spell slots
% Characteristics: Initiative advantage and control abilities, Sneak Attack can make enemies skip turns
ranger(Character) :-
    Weapons   = ['Dagger', 'Shortbow'],
    Inventory = ['Potion of Healing', 'Potion of Healing'],
    Skills    = ['Sneak Attack', 'Hide', 'Dagger Attack', 'Blade Waltz'],
    make_character(ranger, 32, 1, Weapons, Inventory, Skills, 0, Character).

% Paladin: HP 99, AC 10, Apostolic Warhammer, two skills, 3 spell slots
% Easter egg class: Gains overwhelming advantage after transformation, ensures victory in Boss battle
paladin(Character) :-
    Weapons   = ['Apostolic Warhammer'],
    Inventory = [],
    Skills    = ['Warhammer Strike', 'Divine Smite'],
    make_character(paladin, 99, 10, Weapons, Inventory, Skills, 3, Character).

/* ------------------------ Enemy Unit Definitions ------------------------ */

% Imp: HP 6, AC 0, tutorial encounter enemy, relatively weak
imp_enemy(Character) :-
    Weapons   = [],
    Inventory = ['Potion of Healing'],
    Skills    = ['Imp Claw'],
    make_character(imp, 6, 0, Weapons, Inventory, Skills, 0, Character).

% Cambion: HP 40, AC 2, main Boss battle enemy, challenging but beatable
cambion_enemy(Character) :-
    Weapons   = ['Everflame Blade'],
    Inventory = [],
    Skills    = ['Everflame Slash', 'Blood Sacrifice'],
    make_character(cambion, 40, 2, Weapons, Inventory, Skills, 0, Character).

/* ------------------------ Class Attribute Definitions ------------------------ */

% Maximum HP for each class
class_max_hp(fighter, 40).
class_max_hp(wizard, 25).
class_max_hp(ranger, 32).
class_max_hp(paladin, 99).

% Maximum spell slots for each class
class_max_spell_slots(fighter, 0).
class_max_spell_slots(wizard, 3).
class_max_spell_slots(ranger, 0).
class_max_spell_slots(paladin, 3).

% Maximum HP for enemies
enemy_max_hp(imp, 6).
enemy_max_hp(cambion, 40).

% Class to skill tag mapping (used for filtering skills)
class_skill_tag(fighter, fighter_skill).
class_skill_tag(wizard,  wizard_skill).
class_skill_tag(ranger,  ranger_skill).
class_skill_tag(paladin, paladin_skill).

/* ======================================================================
   Skill Modeling and Instances

   All skills are described using a unified skill/10 structure:

     skill(
       Id,          % Unique skill ID (atom)
       Name,        % Display name (string)
       Description, % Skill description (string): detailed effect shown when player selects skill
       Trigger,     % Trigger type: active | on_damaged(Probability) | ...
       ActionCost,  % action_cost(Action, BonusAction, SpellSlots)
       HitRule,     % Hit rule: attack_vs_ac | auto
       DamageSpec,  % Damage info: damage_spec(Dice, FlatMod, DamageType, TargetKind)
       Effects,     % Status effect list: effect(Type, Target, Param, Duration)...
       UsageLimit,  % Usage frequency: unlimited | per_round(N) | per_encounter(N) ...
       Tags         % Other tags: e.g., [melee, weapon_attack, fighter_skill]
     ).

   Skill helper notes:
     - action_cost(Action, BonusAction, SpellSlots) uses 0|1 to indicate resource consumption.
     - hit_rule/1:
         hit_rule(attack_vs_ac) requires attack roll;
         hit_rule(auto) automatically hits.
     - damage_spec(Dice, FlatMod, DamageType, TargetKind):
         Dice uses dice(Num,Sides) format;
         DamageType can be physical/fire/thunder/psychic etc.;
         TargetKind can be enemy | self | all_enemies | allies etc.
     - effect/4 examples:
         effect(bleed, target(enemy), per_round(-1), until(end_of_combat_or_healed)).
         effect(ac_bonus, target(self), amount(5), duration(end_of_combat)).
     - usage_limit/1:
         usage_limit(unlimited) | usage_limit(per_round(N)) | usage_limit(per_encounter(N)).
   ====================================================================== */

/* ------------------------ Fighter Skills ------------------------ */

skill(greatsword_attack,
      'Greatsword Attack',
      'Swing your Greatsword at a designated enemy, dealing 2d6+2 Physical damage. Consumes 1 Action.',
      active,
      action_cost(1, 0, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(2,6), 2, physical, enemy),
      [],
      usage_limit(unlimited),
      [melee, weapon_attack, fighter_skill]).

skill(pommel_strike,
      'Pommel Strike',
      'Strike a designated enemy with the pommel of your sword, dealing 1d4+1 Physical damage. The target is Stunned, losing all actions on their next turn. Consumes 1 Bonus Action. Usable once per encounter.',
      active,
      action_cost(0, 1, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(1,4), 1, physical, enemy),
      [effect(stunned, target(enemy), skip_turn, duration('1_round'))],
      usage_limit(per_encounter(1)),
      [melee, weapon_attack, fighter_skill]).

skill(lacerate,
      'Lacerate',
      'Swing your Greatsword at a designated enemy''s major arteries, dealing 2d6+2 Physical damage. The target suffers from the Bleeding status (loses 2 HP each round until healed). Consumes 1 Action. Usable once per encounter.',
      active,
      action_cost(1, 0, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(2,6), 2, physical, enemy),
      [effect(bleed, target(enemy), per_round(-2), until(healed_or_end_combat))],
      usage_limit(per_encounter(1)),
      [melee, weapon_attack, fighter_skill, bleeding]).

skill(second_wind,
      'Second Wind',
      'Your resolute will tells you it is not yet time to fall. Restore 2d6+3 HP to yourself, and the pain of your wounds lessens. Consumes 1 Bonus Action. Usable once per encounter.',
      active,
      action_cost(0, 1, 0),
      hit_rule(auto),
      damage_spec(dice(2,6), 3, physical, self),
      [],
      usage_limit(per_encounter(1)),
      [self_heal, fighter_skill]).

/* ------------------------ Ranger Skills ------------------------ */

skill(sneak_attack,
      'Sneak Attack',
      'Use your bow to mark all enemies on the field, rapidly shooting an arrow at each one. Deals 2d6+2 Piercing damage to all enemies. Additionally, all enemies in the combat round are Surprised (cannot take any action on their next turn). Consumes no resources. Can only be used on the first round of combat.',
      active,
      action_cost(0, 0, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(2,6), 2, physical, enemy),
      [effect(frightened, target(all_enemies), skip_turn, duration('1_round'))],
      usage_limit(per_round(1)),
      [ranged, bow, ranger_skill, control]).

skill(hide,
      'Hide',
      'Passive. When an enemy makes an attack against you, there is a 30% chance the attack is completely negated (deals 0 damage). Consumes no resources.',
      on_damaged(0.3),
      action_cost(0, 0, 0),
      hit_rule(auto),
      damage_spec(dice(0,0), 0, none, self),
      [],
      usage_limit(unlimited),
      [passive, defensive, ranger_skill]).

skill(dagger_attack,
      'Dagger Attack',
      'Strike a designated enemy with your Dagger, dealing 2d6+1 Piercing damage. Consumes 1 Action.',
      active,
      action_cost(1, 0, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(2,6), 1, physical, enemy),
      [],
      usage_limit(unlimited),
      [melee, weapon_attack, ranger_skill]).

skill(blade_waltz,
      'Blade Waltz',
      'Your Dagger dances in the air, dazzling the enemy. Deal 1d4+1 Piercing damage to a designated target, and the target loses their Action on their next turn. Consumes 1 Bonus Action. Usable once per encounter.',
      active,
      action_cost(0, 1, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(1,4), 1, physical, enemy),
      [effect(confused, target(enemy), action_zero, duration('1_round'))],
      usage_limit(per_encounter(1)),
      [melee, control, ranger_skill]).

/* ------------------------ Wizard Skills ------------------------ */

skill(fire_bolt,
      'Fire Bolt',
      '"Ignis!" A ball of fire forms in your hand and is hurled at the enemy, dealing 2d6+2 Fire damage. Consumes 1 Action.',
      active,
      action_cost(1, 0, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(2,6), 2, fire, enemy),
      [],
      usage_limit(unlimited),
      [ranged, spell, wizard_skill]).

skill(witch_bolt,
      'Witch Bolt',
      '"Per rorare!" A surge of lightning forms in your hand and shoots toward the enemy, dealing 3d8+3 Lightning damage. Consumes 1 Action and 1 Spell Slot.',
      active,
      action_cost(1, 0, 1),
      hit_rule(attack_vs_ac),
      damage_spec(dice(3,8), 3, lightning, enemy),
      [],
      usage_limit(unlimited),
      [spell, wizard_skill]).

skill(thunderwave,
      'Thunderwave',
      '"Tetano!" Unseen sonic waves are summoned and strike the enemies. Deals 2d8+2 Thunder damage to all enemies on the field, and causes them to lose their Action for the current turn. Consumes 1 Action and 1 Spell Slot.',
      active,
      action_cost(1, 0, 1),
      hit_rule(auto),
      damage_spec(dice(2,8), 2, thunder, enemy),
      [effect(shocked, target(all_enemies), action_zero, duration('1_round'))],
      usage_limit(unlimited),
      [spell, control, wizard_skill]).

skill(shield,
      'Shield',
      '"Macteva tute!" You reinforce the surrounding spatial field, increasing your AC by 4 until the end of combat. Consumes 1 Spell Slot and 1 Bonus Action.',
      active,
      action_cost(0, 1, 1),
      hit_rule(auto),
      damage_spec(dice(0,0), 0, none, self),
      [effect(ac_bonus, target(self), amount(4), duration(end_of_combat))],
      usage_limit(unlimited),
      [spell, defensive, wizard_skill]).

/* ------------------------ Paladin Skills ------------------------ */

skill(warhammer_strike,
      'Warhammer Strike',
      'Strike an enemy with your Warhammer, dealing 4d10+10 Physical damage. Consumes 1 Action.',
      active,
      action_cost(1, 0, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(4,10), 10, physical, enemy),
      [],
      usage_limit(unlimited),
      [melee, weapon_attack, paladin_skill]).

skill(divine_smite,
      'Divine Smite',
      'Summon the power of the divine and imbue your Warhammer with radiant energy, striking an enemy for 5d10+20 Radiant damage. Consumes 1 Action and 1 Spell Slot.',
      active,
      action_cost(1, 0, 1),
      hit_rule(attack_vs_ac),
      damage_spec(dice(5,10), 20, radiant, enemy),
      [],
      usage_limit(unlimited),
      [melee, spell, paladin_skill]).

/* ------------------------ Enemy Skills ------------------------ */

skill(imp_claw,
      'Imp Claw',
      'Imp claws at the target, dealing 1d4+1 damage.',
      active,
      action_cost(1, 0, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(1,4), 1, physical, enemy),
      [],
      usage_limit(unlimited),
      [melee, enemy_skill, imp]).

skill(everflame_slash,
      'Everflame Slash',
      'Cambion swings the Everflame Blade to deal 2d6+2 fire damage.',
      active,
      action_cost(1, 0, 0),
      hit_rule(attack_vs_ac),
      damage_spec(dice(2,6), 2, fire, enemy),
      [],
      usage_limit(unlimited),
      [melee, enemy_skill, cambion]).

skill(blood_sacrifice,
      'Blood Sacrifice',
      'Cambion sacrifices its own life force, healing 2d6+3 HP. Usable once per encounter.',
      active,
      action_cost(1, 0, 0),
      hit_rule(auto),
      damage_spec(dice(2,6), 3, healing, self),
      [],
      usage_limit(per_encounter(1)),
      [self_heal, enemy_skill, cambion]).

/* ======================================================================
   Item Modeling and Instances

   Currently only two potion items, represented directly as facts:

     item(Id, Name, Description, EffectTerm)

   In the "use item" logic, effects are resolved based on EffectTerm.
   ====================================================================== */

% Healing Potion: Restores 2d4+2 HP to self
item(potion_of_healing,
     'Potion of Healing',
     'A crimson liquid that knits flesh and bone. Restores 2d4+2 HP to the drinker.',
     heal(self, dice(2,4), 2)).

% Damage Potion: +5 damage to all attacks for the rest of this combat
item(potion_of_damage,
     'Potion of Damage',
     'A volatile concoction that heightens aggression. Grants +5 damage to all your attacks until the end of this combat.',
     buff(self, damage_bonus(5), duration(end_of_combat))).

/* ======================================================================
   Enemy Display Name Definitions
   ====================================================================== */

% Generate display name based on Id (Imp A/B/C)
enemy_display_name(imp1, 'Imp A') :- !.
enemy_display_name(imp2, 'Imp B') :- !.
enemy_display_name(imp3, 'Imp C') :- !.
enemy_display_name(cambion1, 'Cambion Zhalk') :- !.
enemy_display_name(OtherId, Name) :-
    % Fallback: display using Id directly
    format(string(Name), '~w', [OtherId]).

/* ======================================================================
   Numerical Calculation System - Dice and Damage

   Conventions:
     - Dice expression dice(N, S) represents N dice with S sides, e.g., 2d6 -> dice(2,6)
     - Base damage from damage_spec(Dice, FlatMod, DamageType, TargetKind)
         can be calculated via roll_damage/2:
           roll_damage(damage_spec(Dice, Flat, _, _), Total).
   ====================================================================== */

% Roll a single S-sided die, result is between 1..S
roll_die(Sides, Result) :-
    integer(Sides),
    Sides > 0,
    random_between(1, Sides, Result).

% Roll N S-sided dice: dice(N, S), convention is N >= 1
roll_dice(dice(N, Sides), Total) :-
    integer(N),
    N >= 1,
    roll_die(Sides, R1),
    N1 is N - 1,
    ( N1 =:= 0
    -> Rest = 0
    ;  roll_dice(dice(N1, Sides), Rest)
    ),
    Total is R1 + Rest.

% Support multi([...]) format for multiple dice rolls (e.g., 2d8+1d2)
roll_dice(multi(DiceList), Total) :-
    is_list(DiceList),
    roll_dice_list(DiceList, Total).

roll_dice_list([], 0).
roll_dice_list([D|Rest], Total) :-
    roll_dice(D, R1),
    roll_dice_list(Rest, RRest),
    Total is R1 + RRest.

% Calculate final base value from damage_spec (excluding additional buffs)
roll_damage(damage_spec(Dice, FlatMod, _DamageType, _TargetKind), Total) :-
    roll_dice(Dice, DiceValue),
    Total is DiceValue + FlatMod.

/* ======================================================================
   Helper Predicates
   ====================================================================== */

% Remove one element from list (only first occurrence)
remove_one(X, [X|Xs], Xs) :- !.
remove_one(X, [Y|Ys], [Y|Rest]) :-
    remove_one(X, Ys, Rest).

% Check if status is temporary (lasts 1 round)
is_temp_status(stunned).
is_temp_status(frightened).
is_temp_status(confused).
is_temp_status(shocked).

% Check if status is combat-temporary (cleared at end of combat)
is_combat_temp_status(damage_bonus(_)).

/* ======================================================================
   ======================================================================
   ======================================================================
                    GAME FLOW SECTION BELOW
   ======================================================================
   ======================================================================
   ====================================================================== */

/* ======================================================================
   Program Entry and Welcome Message
   ====================================================================== */

/* Automatically display welcome message when program loads */
show_welcome_message :-
    nl,
    write('========================================'), nl,
    write('         Escape from Nautiloid'), nl,
    write('========================================'), nl,
    nl,
    write('Type "start." to begin your adventure.'), nl,
    nl.

:- initialization(show_welcome_message).

/* Game entry point */
start :-
    show_intro,
    nl, nl,
    show_class_selection.

/* ======================================================================
   Generic Typewriter-Style Output
   ====================================================================== */

typing_delay(0).

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

/* ======================================================================
   Opening Narrative and Class Selection
   ====================================================================== */

/* Print opening narrative */
show_intro :-
    typewriter_line('The familiar world vanished in a flash of thunder and alien light. You wake to the cold, pulsating horror of the Nautiloid, a ship forged from flesh and metal.'),
    nl,
    typewriter_line('A searing phantom pain lingers behind your eye --a Illithid Tadpole now resides in your skull, the Mind Flayers'' gruesome gift. You are a captive, a ticking clock, a meal.'),
    nl,
    typewriter_line('But fate has intervened. Shouts and tremors rock the vessel, its organic machinery failing. The Mind Flayer Pod that held you is broken.'),
    nl,
    typewriter_line('The cage is open. Your memory is fractured, but your will to survive is whole. Who were you, before this nightmare began?'),
    nl.

/* Print class selection interface (text display only) */
show_class_selection :-
    typewriter_line('Choose your class:'),
    nl,
    typewriter_line('1. Fighter'),
    typewriter_line('   - High hit points and heavy armor'),
    typewriter_line('   - Starts with a greatsword'),
    typewriter_line('   - Powerful muscles grant extra melee damage'),
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
   Read and Validate Player Class Selection
---------------------------------------------------------------------- */

% Keep reading player input until 1 / 2 / 3 is entered
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

/* Create and record current character based on player input and easter egg */

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

set_player_class(4) :-
    paladin(Char),
    retractall(player_character(_)),
    assertz(player_character(Char)).

/* Get current player character (fails if not set) */
current_player(Char) :-
    player_character(Char).

/* Print complete class info for current player (used during character creation, prompts for confirmation) */
show_player_info(character(Class, HP, AC, Weapons, Inventory, Skills, Action, BonusAction, SpellSlots, StatusEffects)) :-
    display_character_info(character(Class, HP, AC, Weapons, Inventory, Skills, Action, BonusAction, SpellSlots, StatusEffects)),
    ask_confirm_class.

/* Display character info only, without confirmation prompt (used for class transformation etc.) */
display_character_info(character(Class, HP, AC, Weapons, Inventory, Skills, Action, BonusAction, SpellSlots, StatusEffects)) :-
    nl,
    typewriter_line('Your character information:'),
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
    format(string(S9), 'Spell Slots: ~w', [SpellSlots]),
    typewriter_line(S9),
    format(string(S10), 'Status Effects: ~w', [StatusEffects]),
    typewriter_line(S10),
    nl.

/* Ask and confirm whether to keep this class */
ask_confirm_class :-
    typewriter_line('Do you want to keep this class? (y/n)'),
    read(Ans),
    handle_confirm_answer(Ans).

handle_confirm_answer(y) :-
    nl,
    typewriter_line('Class confirmed.'),
    nl, nl,
    show_awakening_scene.

handle_confirm_answer(n) :-
    % Abandon current class, restart selection
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

/* ======================================================================
   Game Scene Display - Awakening from Mind Flayer Pod
   ====================================================================== */

show_awakening_scene :-
    typewriter_line('You stagger free of the shattered Mind Flayer Pod, the cold, viscous remnants of its interior clinging to your skin. Your mind is a haze of throbbing pain and terrifying, alien memories, but the immediate crisis jolts you back to the present. You are aboard the Nautiloid, and it is tearing itself apart.'),
    nl,
    typewriter_line('As you steady yourself to plot your escape, your gaze sweeps the chamber. Surrounding you is a ghastly amphitheater of empty Mind Flayer Pods and, nearby, the grisly corpse of a Mind Flayer itself, which appears to have been killed instantly by a severe blunt impact against the organic wall during the ship\'s recent violent crash.'),
    nl,
    typewriter_line('Dominating the center of the room is a grotesque, pulsating Illithid Tank, filled with shimmering Brine and the sinister, faint outlines of tadpoles.'),
    nl,
    typewriter_line('Beyond the tank, directly across the chamber, a passageway beckons. It is sealed by a thick, wet barrier—a Cranial Valve (or Flesh-Woven Door), the final passage out of this nightmare.'),
    nl,
    show_prison_investigation_options,
    get_prison_investigation_choice.

/* Display list of investigable objects */
show_prison_investigation_options :-
    typewriter_line('The following objects are available for investigation:'),
    nl,
    typewriter_line('1. Mind Flayer Pods'),
    nl,
    typewriter_line('2. Illithid Tank (Brine Pool)'),
    nl,
    typewriter_line('3. Mind Flayer Corpse'),
    nl,
    typewriter_line('4. Cranial Valve (Flesh-Woven Door)'),
    nl,
    typewriter_line('(Enter the corresponding number to investigate:)'),
    nl.

/* ----------------------------------------------------------------------
   Handle Player Investigation Choices in Awakening Scene
---------------------------------------------------------------------- */

get_prison_investigation_choice :-
    read(Choice),
    handle_investigation_choice(Choice).

handle_investigation_choice(1) :-
    investigate_mind_flayer_pods.

handle_investigation_choice(2) :-
    investigate_illithid_tank.

handle_investigation_choice(3) :-
    investigate_mind_flayer_corpse.

handle_investigation_choice(4) :-
    investigate_cranial_valve.

handle_investigation_choice(_) :-
    nl,
    typewriter_line('Invalid input. Please enter 1, 2, 3, or 4.'),
    nl,
    get_prison_investigation_choice.

/* Investigate Mind Flayer Pods */
investigate_mind_flayer_pods :-
    nl,
    typewriter_line('You approach the circle of shattered pods, the air here heavy with residual panic. These are single-person cells, containers built for grotesque experimentation.'),
    nl,
    typewriter_line('The pods themselves are fashioned from hardened, cartilaginous tissue, resembling the thick shell of a beetle or the carapace of a crab. Through the heavy, murky glass, you can glimpse the interior of some: some are empty and scoured clean, while others still hold unidentifiable, dessicated husks—all victims of the same fate you narrowly avoided.'),
    nl,
    typewriter_line('The structure is elongated and chillingly inert now, connecting to the deck by thick, vein-like organic cables, pulsating with dim, internal light. The entire apparatus looks disturbingly like a giant, gaunt squid that has been turned into machinery.'),
    nl,
    typewriter_line('The prisoners are all gone or dead. There is nothing to take here, only a chilling reminder of your escape.'),
    nl, nl,
    show_prison_investigation_options,
    get_prison_investigation_choice.

/* Investigate Illithid Tank */
investigate_illithid_tank :-
    nl,
    typewriter_line('You approach the grotesque Illithid Tank, peering down into the shimmering Brine. The liquid''s surface is slick and unsettling, broken only by the sight of several dead Mind Flayer Tadpoles—the very parasites the Illithids use for ceremorphosis. The sight sends a searing phantom pain through your own eye, an acute reminder of the wriggling horror residing in your skull. You quickly draw back. Survival, not contemplation, must be your single focus now.'),
    nl, nl,
    show_prison_investigation_options,
    get_prison_investigation_choice.

/* Investigate Mind Flayer Corpse */
investigate_mind_flayer_corpse :-
    nl,
    typewriter_line('This is the corpse of an Illithid, or Mind Flayer. The creature stands taller than an average human, but its limbs possess an unsettling, alien grace. Its hands and feet show clear signs of evolutionary atrophy: the hands terminate in four slender, spatulate fingers, while the feet have only two thick, vestigial toes.'),
    nl,
    typewriter_line('The most disturbing feature is its cephalopodian head. The entire cranium is an elongated, ovoid mass of flesh. The highly-developed brain tissue is partially exposed and still twitches faintly—a grotesque post-mortem spasm. Its mouth has degraded into a feeding orifice, surrounded by four long, whip-like tentacles that coil and stiffen in death. The head, in its entirety, unnervingly resembles a giant, dead squid.'),
    nl,
    typewriter_line('A dark, viscous fluid—not blood, but something else—seeps from a fissure in its skull where the severe blunt impact crushed it against the organic wall. This one will not be pursuing you.'),
    nl, nl,
    show_prison_investigation_options,
    get_prison_investigation_choice.

/* Investigate Cranial Valve */
investigate_cranial_valve :-
    cranial_valve_state(closed),
    nl,
    typewriter_line('You approach the fleshy barrier sealing the exit. It is a thick, pulsating membrane, glistening with a mixture of amniotic fluid and something slicker. This is the valve that controls the passage to the rest of the ship. It is stitched together with taut, sinewy tissue, resembling a massive, perpetually contracting sphincter.'),
    nl,
    typewriter_line('You can feel a strange, muffled pressure building from the other side, hinting at the vast, organic network of the Nautiloid waiting beyond. It doesn''t appear to be damaged by the crash, only tightly sealed.'),
    nl,
    ask_open_valve.

investigate_cranial_valve :-
    cranial_valve_state(open),
    ask_enter_next_room.

/* Ask player whether to open the door */
ask_open_valve :-
    nl,
    typewriter_line('Do you risk touching the Cranial Valve to force it open and proceed? (y/n)'),
    nl,
    read(Answer),
    handle_valve_answer(Answer).

handle_valve_answer(y) :-
    retractall(cranial_valve_state(_)),
    assertz(cranial_valve_state(open)),
    nl,
    typewriter_line('You reach out and touch the pulsating membrane. With a wet, organic sound, the Cranial Valve begins to contract and open, revealing the passage beyond.'),
    nl,
    show_surgical_theater_scene.

handle_valve_answer(n) :-
    nl,
    typewriter_line('You decide not to risk it for now.'),
    nl, nl,
    show_prison_investigation_options,
    get_prison_investigation_choice.

handle_valve_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_open_valve.

/* Ask whether to enter the next room (when door is already open) */
ask_enter_next_room :-
    nl,
    typewriter_line('The Cranial Valve is already opened. Do you want to enter the next room? (y/n)'),
    nl,
    read(Answer),
    handle_enter_next_room_answer(Answer).

handle_enter_next_room_answer(y) :-
    nl,
    typewriter_line('You enter the next room.'),
    show_surgical_theater_options.

handle_enter_next_room_answer(n) :-
    nl,
    typewriter_line('You decide to stay in the current chamber.'),
    nl, nl,
    show_prison_investigation_options,
    get_prison_investigation_choice.

handle_enter_next_room_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_enter_next_room.

/* ======================================================================
   Game Scene Display - Surgical Theater
   ====================================================================== */

show_surgical_theater_scene :-
    nl,
    typewriter_line('You pass through the Cranial Valve and find yourself in a vast, semicircular chamber—a grotesque surgical theater.'),
    nl,
    typewriter_line('The overhead Brine Lamps cast an ominous, pulsating red glow across the scene. A line of Operating Tables stands along the periphery, most of them empty, save for two: one bears the small, contorted corpse of a Goblin, and the other, the slender, pale body of a High Elf.'),
    nl,
    typewriter_line('All around the chamber are unsettling arrangements of large, glowing glass Canisters filled with strange fluids, emitting that same unsettling red light. Next to these stands a bizarre Slate, carved from a material you cannot identify.'),
    nl,
    typewriter_line('At the far end of the room, a high Archway gapes open, offering access to the next, unknown area of the Nautiloid.'),
    nl,
    show_surgical_theater_options.

/* Display list of investigable objects in Surgical Theater */
show_surgical_theater_options :-
    typewriter_line('The following objects are available for investigation:'),
    nl,
    typewriter_line('1. Cranial Valve (Return to previous chamber)'),
    nl,
    typewriter_line('2. High Elf Corpse'),
    nl,
    typewriter_line('3. Goblin Corpse'),
    nl,
    typewriter_line('4. Glowing Canisters'),
    nl,
    typewriter_line('5. Mysterious Slate'),
    nl,
    typewriter_line('6. High Archway'),
    nl,
    typewriter_line('(Enter the corresponding number to investigate:)'),
    nl,
    get_surgical_theater_choice.

/* ----------------------------------------------------------------------
   Handle Player Investigation Choices in Surgical Theater
---------------------------------------------------------------------- */

get_surgical_theater_choice :-
    read(Choice),
    handle_surgical_theater_choice(Choice).

handle_surgical_theater_choice(1) :-
    ask_return_previous_chamber.

handle_surgical_theater_choice(2) :-
    investigate_high_elf_corpse.

handle_surgical_theater_choice(3) :-
    investigate_goblin_corpse.

handle_surgical_theater_choice(4) :-
    investigate_glowing_canisters.

handle_surgical_theater_choice(5) :-
    investigate_mysterious_slate.

handle_surgical_theater_choice(6) :-
    investigate_high_archway.

handle_surgical_theater_choice(_) :-
    nl,
    typewriter_line('Invalid input. Please enter 1, 2, 3, 4, 5, or 6.'),
    nl,
    get_surgical_theater_choice.

/* Ask whether to return to previous chamber */
ask_return_previous_chamber :-
    nl,
    typewriter_line('You will return to the previous chamber. Do you want to proceed? (y/n)'),
    nl,
    read(Answer),
    handle_return_answer(Answer).

handle_return_answer(y) :-
    nl,
    typewriter_line('You pass back through the Cranial Valve to the previous chamber.'),
    nl, nl,
    show_prison_investigation_options,
    get_prison_investigation_choice.

handle_return_answer(n) :-
    nl,
    typewriter_line('You decide to stay in the surgical theater.'),
    nl, nl,
    show_surgical_theater_options.

handle_return_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_return_previous_chamber.

/* Investigate High Elf Corpse */
investigate_high_elf_corpse :-
    nl,
    typewriter_line('The High Elf''s corpse is pale and remarkably preserved, retaining the sharp, high cheekbones and pointed ears characteristic of their race. They are dressed in simple, but finely woven robes—unfitting for a prisoner.'),
    nl,
    typewriter_line('However, the elegant features are overshadowed by the brutal experiment performed upon them. The skull has been precisely opened—a terrifyingly clean incision suggesting a highly refined, surgical procedure—revealing the interior. The chamber that should contain the brain is entirely empty, a hollow, crimson cavity.'),
    nl,
    typewriter_line('This was not meant to kill them, but to harvest or replace their most vital organ. This chilling efficiency leaves no physical evidence to gather.'),
    nl, nl,
    show_surgical_theater_options,
    get_surgical_theater_choice.

/* Investigate Goblin Corpse */
investigate_goblin_corpse :-
    nl,
    typewriter_line('The small, huddled Goblin corpse lies on the operating table. Its skin is a rough, sickly grey-green, and its small limbs are contorted into a painful, unnatural position.'),
    nl,
    typewriter_line('A horrifying surgical procedure was performed upon this subject. The small cranium has been precisely opened—a terrifyingly clean incision suggesting a highly refined, cold surgical procedure. The chamber inside the skull is completely empty, a hollow, crimson cavity where the brain should reside.'),
    nl,
    typewriter_line('This demonstrates the Mind Flayers'' standardized and chillingly efficient process for harvesting their subjects, regardless of the target''s species.'),
    nl, nl,
    show_surgical_theater_options,
    get_surgical_theater_choice.

/* Investigate Glowing Canisters */
investigate_glowing_canisters :-
    nl,
    typewriter_line('The Canister is a thick glass cylinder, pulsating with an eerie red glow. It is filled to the brim with a clear, viscous fluid—likely a form of nutrient solution—through which continuous streams of bubbles ascend from the base.'),
    nl,
    typewriter_line('You press closer to the glass. Inside the fluid, suspended in the faint light, is a fully intact brain. It appears disturbingly fresh and, incredibly, still shows faint, rhythmic twitches—suggesting the mind within it is still tragically alive.'),
    nl,
    typewriter_line('This is a storage unit for harvested organs. The glass is too thick to break without a heavy weapon, and there is no visible mechanism to interact with the container.'),
    nl, nl,
    show_surgical_theater_options,
    get_surgical_theater_choice.

/* Investigate Mysterious Slate */
investigate_mysterious_slate :-
    nl,
    typewriter_line('The Slate is carved from a dark, unusual stone, its surface etched with complex, glowing glyphs that belong to no known language. They feel disturbingly familiar, however, resonating with the parasite in your skull.'),
    nl,
    typewriter_line('You reach out and touch the stone. Instantly, your consciousness is plunged into a jarring silence. A raw, chaotic torrent of knowledge—not yours, but borrowed from the Mind Flayers'' network—floods your mind:'),
    nl,
    typewriter_line('The brutal, chaotic hunting scenes of the Goblins in the Underdark.'),
    nl,
    typewriter_line('A flash of the High Elf''s refined history and arcane culture, spanning centuries.'),
    nl,
    typewriter_line('The complex, militant research findings of the Githyanki—warriors who dedicate their lives to slaying Mind Flayers.'),
    nl,
    typewriter_line('The influx is overwhelming, a powerful psychic overload that leaves your head throbbing. You instinctively pull your hand away, reeling from the vast, unsettling scope of data that has just been forcibly imprinted onto your mind.'),
    nl, nl,
    show_surgical_theater_options,
    get_surgical_theater_choice.

/* Investigate High Archway */
investigate_high_archway :-
    nl,
    typewriter_line('You pass through the High Archway. The passage curves sharply, and you realize the corridor is utterly exposed—the hull has been ripped away by the fierce celestial assault. It is here that the terrifying truth of your location is revealed: the Nautiloid has been warped to Avernus, the first layer of the Nine Hells.'),
    nl,
    typewriter_line('The air is acrid and hot. Your gaze sweeps over the landscape: barren, jutting mountains loom over blood-red earth, and colossal boulders etched with sinister runes float unnaturally in the infernal sky. Swarms of tiny, winged Imps (small devils) continuously circle and descend upon the damaged ship, clawing at its flesh.'),
    nl,
    typewriter_line('Although your instinct screams to abandon the Nautiloid immediately, a leap from this height, directly into the chaos below, would be certain death. You must continue deeper into the ship. Following the bend of the corridor, you see the entrance to another room standing wide open.'),
    nl, nl,
    ask_enter_next_area_from_archway.

ask_enter_next_area_from_archway :-
    typewriter_line('Do you want to proceed into the next room? (y/n)'),
    nl,
    read(Answer),
    handle_enter_next_area_answer(Answer).

handle_enter_next_area_answer(y) :-
    nl,
    typewriter_line('You steel yourself and move on toward the next chamber...'),
    nl, nl,
    enter_next_combat_room,
    !.  % Ensure no backtracking after entering combat

handle_enter_next_area_answer(n) :-
    nl,
    typewriter_line('You step back from the exposed corridor, returning to the surgical theater.'),
    nl, nl,
    show_surgical_theater_options,
    get_surgical_theater_choice.

handle_enter_next_area_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_enter_next_area_from_archway.

/* ======================================================================
   Enter Combat Room and Combat Initialization
   ====================================================================== */

enter_next_combat_room :-
    % Initialize this combat: spawn 3 imp enemies
    setup_imp_encounter,
    ( current_player(character(ranger, _, _, _, _, _, _, _, _, _))
    -> nl,
       typewriter_line('Your heightened Ranger Senses immediately alert you to danger ahead. You slow your pace and move forward with practiced stealth. As you round the bend, the threat is clear: three small Imps are clustered around a discarded prisoner''s corpse, tearing at the flesh in a grotesque feast. They are entirely engrossed in their meal and have not registered your presence. This is a perfect opportunity to exploit their distraction and gain the advantage of surprise.'),
       nl
    ;  nl,
       typewriter_line('You continue your advance, moving boldly along the exposed corridor and quickly stepping into the next chamber. Your stride abruptly halts as you gasp: three small Imps are clustered, greedily tearing at the remains of a recently deceased prisoner. The gruesome sight fills you with a mix of shock and righteous fury. Disturbed from their feast, the small devils swivel their grotesque heads, letting out sharp, hostile screeches directed at you. They are eager for a new meal. The opportunity for surprise is gone. Roll for Initiative!'),
       nl
    ),
    start_combat_loop,
    !.  % Ensure no backtracking after combat loop ends

% Initialize a combat encounter with three imps
setup_imp_encounter :-
    retractall(enemy_instance(_, _)),
    retractall(current_combat(_)),
    assertz(current_combat(imp_encounter)),
    imp_enemy(ImpChar1),
    imp_enemy(ImpChar2),
    imp_enemy(ImpChar3),
    assertz(enemy_instance(imp1, ImpChar1)),
    assertz(enemy_instance(imp2, ImpChar2)),
    assertz(enemy_instance(imp3, ImpChar3)).

/* Initialize Boss battle encounter */
setup_boss_encounter :-
    retractall(enemy_instance(_, _)),
    retractall(current_combat(_)),
    assertz(current_combat(boss_encounter)),
    cambion_enemy(CambionChar),
    imp_enemy(ImpChar1),
    imp_enemy(ImpChar2),
    assertz(enemy_instance(cambion1, CambionChar)),
    assertz(enemy_instance(imp1, ImpChar1)),
    assertz(enemy_instance(imp2, ImpChar2)).

/* ======================================================================
   Combat Round System - Player goes first, then enemies, until one side is defeated
   ====================================================================== */

% Main combat loop skeleton
start_combat_loop :-
    retractall(combat_round(_)),
    assertz(combat_round(1)),
    retractall(skill_usage_count(_, _)),
    combat_loop.

combat_loop :-
    ( combat_over(player_dead)
    -> handle_combat_end(player_dead)
    ;  combat_over(enemies_dead)
    -> handle_combat_end(enemies_dead)
    ;  player_turn,
       ( combat_over(player_dead)
       -> handle_combat_end(player_dead)
       ;  combat_over(enemies_dead)
       -> handle_combat_end(enemies_dead)
       ;  enemies_turn,
          increment_combat_round,
          combat_loop
       )
    ).

% Increment round counter
increment_combat_round :-
    retract(combat_round(N)),
    N1 is N + 1,
    assertz(combat_round(N1)),
    % Reset Sneak Attack per-round counter
    retractall(skill_usage_count(sneak_attack, used_this_round)),
    % Reset player's Action and Bonus Action
    current_player(character(Class, HP, AC, W, Inv, Skills, _, _, SpellSlots, Status)),
    retractall(player_character(_)),
    assertz(player_character(character(Class, HP, AC, W, Inv, Skills, 1, 1, SpellSlots, Status))),
    % Clear all enemy temporary status effects (stunned, frightened, confused, shocked)
    clear_all_enemy_temp_status.

% Clear all enemy temporary status effects
clear_all_enemy_temp_status :-
    findall(Id, enemy_instance(Id, _), EnemyIds),
    clear_each_enemy_temp_status(EnemyIds).

clear_each_enemy_temp_status([]) :- !.
clear_each_enemy_temp_status([EnemyId|Rest]) :-
    enemy_instance(EnemyId, character(EClass, HP, AC, W, Inv, Skills, _, _, Slots, Status)),
    % Remove temporary status: stunned, frightened, confused, shocked
    exclude(is_temp_status, Status, NewStatus),
    retractall(enemy_instance(EnemyId, _)),
    % Reset enemy's Action and Bonus Action to 1
    assertz(enemy_instance(EnemyId, character(EClass, HP, AC, W, Inv, Skills, 1, 1, Slots, NewStatus))),
    clear_each_enemy_temp_status(Rest).

% Determine if combat is over
combat_over(player_dead) :-
    current_player(character(_, HP, _, _, _, _, _, _, _, _)),
    HP =< 0,
    !.

combat_over(enemies_dead) :-
    \+ ( enemy_instance(_, character(_, HP, _, _, _, _, _, _, _, _)),
         HP > 0 ),
    !.

% Combat end handling: return to exploration or show defeat screen
handle_combat_end(player_dead) :-
    nl,
    typewriter_line('Your vision darkens as you collapse to the fleshy floor of the Nautiloid...'),
    typewriter_line('You have been slain. The imps continue their grisly feast, uncaring.'),
    nl,
    typewriter_line('GAME OVER.'),
    nl,
    !.  % Ensure no backtracking

handle_combat_end(enemies_dead) :-
    % Clear combat-end temporary status
    clear_combat_temp_status,
    nl,
    ( current_combat(boss_encounter)
    -> typewriter_line('With a final, agonized roar, the Cambion Zhalk collapses, its flaming blade clattering to the floor. The remaining imps shriek in terror and flee into the shadows. Victory is yours.'),
       nl,
       show_boss_victory_scene
    ;  typewriter_line('The last imp falls, its body dissolving into foul ichor. The chamber falls eerily silent.'),
       nl,
       show_post_combat_scene
    ),
    !.  % Ensure no backtracking

% Clear combat-end temporary status
clear_combat_temp_status :-
    current_player(character(Class, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status)),
    % Remove damage_bonus status
    exclude(is_combat_temp_status, Status, NewStatus),
    retractall(player_character(_)),
    assertz(player_character(character(Class, HP, AC, W, Inv, Skills, Act, BAct, Slots, NewStatus))).

/* ======================================================================
   Player Turn - Display Status and Action Selection
   ====================================================================== */

player_turn :-
    nl,
    combat_round(Round),
    format(string(RoundMsg), '--- Round ~w: Player Turn ---', [Round]),
    typewriter_line(RoundMsg),
    player_action_loop.

% Repeat actions within a player turn until player chooses to end turn
player_action_loop :-
    show_battle_status,
    nl,
    typewriter_line('Choose your action:'),
    typewriter_line('1. Use Skill'),
    typewriter_line('2. Use Item'),
    typewriter_line('3. End Turn'),
    nl,
    read(Choice),
    handle_player_choice(Choice).

handle_player_choice(1) :-
    nl,
    typewriter_line('Available skills:'),
    show_available_skills(SkillList),
    length(SkillList, SkillCount),
    nl,
    typewriter_line('Please select a skill number to use, or enter 0 to cancel:'),
    nl,
    read(SkillChoice),
    ( SkillChoice = 0
    -> nl,
       typewriter_line('Action cancelled. Returning to action selection...'),
       nl,
       player_action_loop
    ;  ( integer(SkillChoice),
          SkillChoice >= 1, SkillChoice =< SkillCount,
         nth1(SkillChoice, SkillList, SelectedSkillId)
       -> nl,
          resolve_skill_use(SelectedSkillId)
       ;  nl,
          typewriter_line('Invalid skill number. Please try again.'),
          nl,
          handle_player_choice(1)
       )
    ).

handle_player_choice(2) :-
    nl,
    typewriter_line('Available items (each item consumes 1 Bonus Action):'),
    show_available_items(ItemList),
    length(ItemList, ItemCount),
    ( ItemList = []
    -> nl,
       typewriter_line('No items available. Returning to action selection...'),
       nl,
       player_action_loop
    ;  nl,
       typewriter_line('Please select an item number to use, or enter 0 to cancel:'),
       nl,
       read(ItemChoice),
       ( ItemChoice = 0
       -> nl,
          typewriter_line('Action cancelled. Returning to action selection...'),
          nl,
          player_action_loop
       ;  ( integer(ItemChoice),
            ItemChoice >= 1, ItemChoice =< ItemCount,
            nth1(ItemChoice, ItemList, SelectedItemName)
          -> nl,
             resolve_item_use(SelectedItemName),
             nl,
             player_action_loop
          ;  nl,
             typewriter_line('Invalid item number. Please try again.'),
             nl,
             handle_player_choice(2)
          )
       )
    ).

handle_player_choice(3) :-
    nl,
    typewriter_line('Do you want to end your turn? (y/n)'),
    nl,
    read(Ans),
    ( Ans = y
    -> nl,
       typewriter_line('You end your turn.'),
       nl
    ;  Ans = n
    -> nl,
       typewriter_line('You decide to keep fighting this round.'),
       nl,
       player_action_loop
    ;  nl,
       typewriter_line('Please answer with y or n.'),
       nl,
       handle_player_choice(3)
    ).

handle_player_choice(_) :-
    nl,
    typewriter_line('Invalid choice. Please enter 1, 2, or 3.'),
    nl,
    player_action_loop.

/* ======================================================================
   Skill Target and Casting Flow
   ====================================================================== */

% Determine if skill primarily targets self
skill_targets_self(SkillId) :-
    skill(SkillId, _, _, _, _, _, damage_spec(_, _, _, self), _, _, _),
    !.
skill_targets_self(SkillId) :-
    % Or Effects explicitly contain target(self)
    skill(SkillId, _, _, _, _, _, _, Effects, _, _),
    member(effect(_, target(self), _, _), Effects),
    !.

% Check if skill targets enemy (simple check: DamageSpec target is enemy or all_enemies)
skill_targets_enemy(SkillId) :-
    skill(SkillId, _, _, _, _, _, damage_spec(_, _, _, enemy), _, _, _),
    !.
skill_targets_enemy(SkillId) :-
    skill(SkillId, _, _, _, _, _, damage_spec(_, _, _, all_enemies), _, _, _),
    !.

% Determine if skill targets all enemies
skill_targets_all_enemies(SkillId) :-
    skill(SkillId, _, _, _, _, _, _, Effects, _, _),
    member(effect(_, target(all_enemies), _, _), Effects),
    !.

% Record skill usage
record_skill_usage(sneak_attack) :-
    !,
    assertz(skill_usage_count(sneak_attack, used_this_round)).

record_skill_usage(SkillId) :-
    skill(SkillId, _, _, _, _, _, _, _, usage_limit(per_encounter(1)), _),
    !,
    assertz(skill_usage_count(SkillId, used)).

record_skill_usage(_).

% Determine subsequent flow based on skill type
resolve_skill_use(SkillId) :-
    skill(SkillId, _, _, _, _, _, _, _, UsageLimit, _),
    ( check_skill_usage_allowed(SkillId, UsageLimit)
    -> ( skill_targets_self(SkillId)
       -> ask_use_self_skill(SkillId)
       ;  skill_targets_all_enemies(SkillId)
       -> ask_use_all_enemies_skill(SkillId)
       ;  ask_select_enemy_target(SkillId)
       )
    ;  nl,
       ( SkillId = sneak_attack
       -> typewriter_line('Sneak Attack can only be used once on the first round of combat.')
       ;  typewriter_line('This skill has already been used and cannot be used again in this encounter.')
       ),
       nl,
       handle_player_choice(1)
    ).

% Check if skill can be used
check_skill_usage_allowed(SkillId, usage_limit(per_encounter(1))) :-
    !,
    \+ skill_usage_count(SkillId, used).

check_skill_usage_allowed(sneak_attack, usage_limit(per_round(1))) :-
    !,
    combat_round(Round),
    Round =:= 1,
    \+ skill_usage_count(sneak_attack, used_this_round).

check_skill_usage_allowed(_, _) :-
    true.

/* ---------------- Self-targeting Skills: Confirm use on self ---------------- */

ask_use_self_skill(SkillId) :-
    skill(SkillId, Name, _, _, _, _, _, _, _, _),
    format(string(Msg), 'This skill (~w) targets yourself. Do you want to use it on yourself? (y/n)', [Name]),
    typewriter_line(Msg),
    nl,
    read(Ans),
    handle_self_skill_confirm(SkillId, Ans).

handle_self_skill_confirm(SkillId, y) :-
    apply_self_skill_effect(SkillId),
    nl,
    player_action_loop.

handle_self_skill_confirm(_, n) :-
    nl,
    typewriter_line('Cancelled. Returning to skill selection...'),
    nl,
    handle_player_choice(1).

handle_self_skill_confirm(SkillId, _) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_use_self_skill(SkillId).

/* ---------------- Enemy-targeting Skills: Select target enemy ---------------- */

% Get list of currently alive enemy IDs (sorted in fixed order for consistent display)
alive_enemy_ids(EnemyIds) :-
    findall(Id,
            ( enemy_instance(Id, character(_, HP, _, _, _, _, _, _, _, _)),
              HP > 0
            ),
            Ids0),
    sort(Ids0, EnemyIds).

% Cast skill on all enemies
ask_use_all_enemies_skill(SkillId) :-
    skill(SkillId, Name, _, _, _, _, _, _, _, _),
    format(string(Msg), 'You are about to use "~w" on all enemies. Proceed? (y/n)', [Name]),
    typewriter_line(Msg),
    nl,
    read(Ans),
    handle_all_enemies_skill_confirm(SkillId, Ans).

handle_all_enemies_skill_confirm(SkillId, y) :-
    apply_all_enemies_skill_effect(SkillId),
    nl,
    player_action_loop.

handle_all_enemies_skill_confirm(_, n) :-
    nl,
    typewriter_line('Skill cancelled. Returning to skill selection...'),
    nl,
    handle_player_choice(1).

handle_all_enemies_skill_confirm(SkillId, _) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_use_all_enemies_skill(SkillId).

ask_select_enemy_target(SkillId) :-
    alive_enemy_ids(EnemyIds),
    ( EnemyIds = []
    -> nl,
       typewriter_line('There are no valid enemy targets. Returning to action selection...'),
       nl,
       player_turn
    ;  nl,
       typewriter_line('Choose a target for this skill:'),
       show_enemy_targets_numbered(EnemyIds, 1),
       nl,
       typewriter_line('Enter the enemy number to target, or 0 to go back:'),
       nl,
       read(TargetChoice),
       handle_enemy_target_choice(SkillId, TargetChoice, EnemyIds)
    ).

% Display enemy list with numbers
show_enemy_targets_numbered([], _).
show_enemy_targets_numbered([Id|Rest], N) :-
    enemy_instance(Id, character(EClass, HP, AC, _, _, _, _, _, _, Status)),
    enemy_display_name(Id, DisplayName),
    format(string(S), '~w. ~w (~w) HP: ~w, AC: ~w, Status: ~w',
           [N, DisplayName, EClass, HP, AC, Status]),
    typewriter_line(S),
    N1 is N + 1,
    show_enemy_targets_numbered(Rest, N1).

% Handle target selection input
handle_enemy_target_choice(_SkillId, 0, _EnemyIds) :-
    nl,
    typewriter_line('Cancelled. Returning to skill selection...'),
    nl,
    handle_player_choice(1).

handle_enemy_target_choice(SkillId, TargetChoice, EnemyIds) :-
    length(EnemyIds, Count),
    ( integer(TargetChoice),
      TargetChoice >= 1,
      TargetChoice =< Count
    -> nth1(TargetChoice, EnemyIds, TargetId),
       confirm_attack_target(SkillId, TargetId)
    ;  nl,
       typewriter_line('Invalid target number. Please try again.'),
       nl,
       ask_select_enemy_target(SkillId)
    ).

% Confirm whether to attack selected enemy
confirm_attack_target(SkillId, TargetId) :-
    skill(SkillId, Name, _, _, _, _, _, _, _, _),
    enemy_display_name(TargetId, DisplayName),
    format(string(Msg), 'You are about to use "~w" on ~w. Proceed? (y/n)',
           [Name, DisplayName]),
    typewriter_line(Msg),
    nl,
    read(Ans),
    handle_attack_confirm(SkillId, TargetId, Ans).

handle_attack_confirm(SkillId, TargetId, y) :-
    apply_attack_skill_effect(SkillId, TargetId),
    nl,
    player_action_loop.

handle_attack_confirm(SkillId, _TargetId, n) :-
    nl,
    typewriter_line('Attack cancelled. Returning to target selection...'),
    nl,
    ask_select_enemy_target(SkillId).

handle_attack_confirm(SkillId, TargetId, _) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    confirm_attack_target(SkillId, TargetId).

/* ======================================================================
   Skill and Item Effect Resolution
   ====================================================================== */

% Deduct resources: Action/BonusAction/SpellSlots are all 0/1 flags
pay_action_cost(
    character(Class, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status),
    action_cost(CA, CB, CS),
    character(Class, HP, AC, W, Inv, Skills, NewAct, NewBAct, NewSlots, Status)
) :-
    Act  >= CA,
    BAct >= CB,
    Slots >= CS,
    NewAct   is Act  - CA,
    NewBAct  is BAct - CB,
    NewSlots is Slots - CS.

% Cast skill on self (e.g., Second Wind, Shield)
apply_self_skill_effect(SkillId) :-
    skill(SkillId, Name, _, _, ActionCost, _, DamageSpec, Effects, _, Tags),
    current_player(Char0),
    ( pay_action_cost(Char0, ActionCost, Char1)
    -> Char1 = character(Class, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status),
       % Only skills with self_heal tag perform healing (e.g., Second Wind)
       ( member(self_heal, Tags)
       -> roll_damage(DamageSpec, Amount),
          NewHP is HP + Amount,
          HealMsg = true
       ;  Amount = 0,
          NewHP = HP,
          HealMsg = false
       ),
       % Handle AC bonus effect (e.g., Shield)
       ( member(effect(ac_bonus, target(self), amount(Bonus), _), Effects)
       -> NewAC is AC + Bonus,
          NewStatus = [ac_bonus(Bonus)|Status]
       ;  NewAC = AC,
          NewStatus = Status
       ),
       retractall(player_character(_)),
       assertz(player_character(character(Class, NewHP, NewAC, W, Inv, Skills, Act, BAct, Slots, NewStatus))),
       record_skill_usage(SkillId),
       % Display different messages based on skill type
       ( HealMsg = true
       -> format(string(Msg), 'You use ~w on yourself and recover ~w HP.', [Name, Amount])
       ;  format(string(Msg), 'You use ~w on yourself.', [Name])
       ),
       typewriter_line(Msg),
       nl
    ;  typewriter_line('Not enough resources to use this skill.'),
       nl
    ).

% Apply status effects to target
apply_status_effects([], _, Status, Status) :- !.

apply_status_effects([effect(StatusType, target(TargetKind), Param, Duration)|Rest], TargetId, Status, NewStatus) :-
    !,
    ( TargetKind = enemy
    -> apply_single_status(StatusType, Param, Duration, Status, TempStatus)
    ;  TempStatus = Status
    ),
    apply_status_effects(Rest, TargetId, TempStatus, NewStatus).

apply_status_effects([effect(StatusType, target(all_enemies), Param, Duration)|Rest], TargetId, Status, NewStatus) :-
    !,
    apply_single_status(StatusType, Param, Duration, Status, TempStatus),
    apply_status_effects(Rest, TargetId, TempStatus, NewStatus).

apply_status_effects([_|Rest], TargetId, Status, NewStatus) :-
    apply_status_effects(Rest, TargetId, Status, NewStatus).

% Apply single status effect
apply_single_status(stunned, skip_turn, duration('1_round'), Status, [stunned|Status]) :- !.
apply_single_status(bleed, per_round(_), until(healed_or_end_combat), Status, [bleeding|Status]) :- !.
apply_single_status(frightened, skip_turn, duration('1_round'), Status, [frightened|Status]) :- !.
apply_single_status(confused, action_zero, duration('1_round'), Status, [confused|Status]) :- !.
apply_single_status(shocked, action_zero, duration('1_round'), Status, [shocked|Status]) :- !.
apply_single_status(_, _, _, Status, Status).

% Cast skill on enemy: roll dice based on damage_spec, subtract enemy AC for final damage
apply_attack_skill_effect(SkillId, TargetId) :-
    skill(SkillId, Name, _, _, ActionCost, _, DamageSpec, Effects, _, _),
    current_player(Char0),
    ( pay_action_cost(Char0, ActionCost, CharAfterCost)
    -> % Update player resources
       CharAfterCost = character(Class, PHP, PAC, PW, PInv, PSkills, PAct, PBAct, PSlots, PStatus),
       retractall(player_character(_)),
       assertz(player_character(character(Class, PHP, PAC, PW, PInv, PSkills, PAct, PBAct, PSlots, PStatus))),
       % Calculate base damage
       roll_damage(DamageSpec, Base),
       % Calculate extra damage (if there's an extra_damage effect)
       calculate_extra_damage(Effects, ExtraDamage),
       % Calculate damage bonus buff (if player has damage_bonus status)
       ( member(damage_bonus(Bonus), PStatus)
       -> DamageBonus = Bonus
       ;  DamageBonus = 0
       ),
       TotalBase is Base + ExtraDamage + DamageBonus,
       enemy_instance(TargetId, character(EClass, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status)),
       DamageRaw is TotalBase - AC,
       ( DamageRaw =< 0 -> Damage is 0 ; Damage is DamageRaw ),
       NewHP is max(0, HP - Damage),
       % Apply status effects
       apply_status_effects(Effects, TargetId, Status, NewStatus),
       retractall(enemy_instance(TargetId, _)),
       assertz(enemy_instance(TargetId, character(EClass, NewHP, AC, W, Inv, Skills, Act, BAct, Slots, NewStatus))),
       record_skill_usage(SkillId),
       % Handle all-enemies status effects (only when there's a target(all_enemies) effect)
       ( has_all_enemies_effect(Effects)
       -> apply_all_enemies_status(Effects)
       ;  true
       ),
       enemy_display_name(TargetId, DisplayName),
       format(string(Msg), 'You use ~w on ~w and deal ~w damage (after AC).', [Name, DisplayName, Damage]),
       typewriter_line(Msg),
       nl
    ;  typewriter_line('Not enough resources to use this skill.'),
       nl
    ).

% Calculate extra damage (handle extra_damage effect)
calculate_extra_damage([], 0) :- !.
calculate_extra_damage([effect(extra_damage, target(enemy), DiceSpec, instant)|Rest], Total) :-
    !,
    roll_dice(DiceSpec, ExtraDmg),
    calculate_extra_damage(Rest, RestDmg),
    Total is ExtraDmg + RestDmg.
calculate_extra_damage([_|Rest], Total) :-
    calculate_extra_damage(Rest, Total).

% Check if there's a target(all_enemies) effect
has_all_enemies_effect(Effects) :-
    member(effect(_, target(all_enemies), _, _), Effects),
    !.

% Apply status effects to all enemies (only target(all_enemies) effects)
apply_all_enemies_status(Effects) :-
    findall(Id, enemy_instance(Id, _), EnemyIds),
    apply_status_to_all_enemies(Effects, EnemyIds).

apply_status_to_all_enemies(_, []) :- !.
apply_status_to_all_enemies(Effects, [EnemyId|Rest]) :-
    enemy_instance(EnemyId, character(EClass, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status)),
    % Only apply target(all_enemies) effects
    apply_all_enemies_effects(Effects, Status, NewStatus),
    retractall(enemy_instance(EnemyId, _)),
    assertz(enemy_instance(EnemyId, character(EClass, HP, AC, W, Inv, Skills, Act, BAct, Slots, NewStatus))),
    apply_status_to_all_enemies(Effects, Rest).

% Only apply target(all_enemies) effects
apply_all_enemies_effects([], Status, Status) :- !.
apply_all_enemies_effects([effect(StatusType, target(all_enemies), Param, Duration)|Rest], Status, NewStatus) :-
    !,
    apply_single_status(StatusType, Param, Duration, Status, TempStatus),
    apply_all_enemies_effects(Rest, TempStatus, NewStatus).
apply_all_enemies_effects([_|Rest], Status, NewStatus) :-
    apply_all_enemies_effects(Rest, Status, NewStatus).

% Apply skill effects to all enemies
apply_all_enemies_skill_effect(SkillId) :-
    skill(SkillId, Name, _, _, ActionCost, _, DamageSpec, Effects, _, _),
    current_player(Char0),
    ( pay_action_cost(Char0, ActionCost, CharAfterCost)
    -> % Update player resources
       CharAfterCost = character(Class, PHP, PAC, PW, PInv, PSkills, PAct, PBAct, PSlots, PStatus),
       retractall(player_character(_)),
       assertz(player_character(character(Class, PHP, PAC, PW, PInv, PSkills, PAct, PBAct, PSlots, PStatus))),
       % Apply damage and status to all enemies
       findall(Id, enemy_instance(Id, _), EnemyIds),
       apply_skill_to_all_enemies(SkillId, Name, DamageSpec, Effects, EnemyIds),
       record_skill_usage(SkillId)
    ;  typewriter_line('Not enough resources to use this skill.'),
       nl
    ).

% Apply skill to each enemy
apply_skill_to_all_enemies(_, _, _, _, []) :- !.
apply_skill_to_all_enemies(SkillId, Name, DamageSpec, Effects, [EnemyId|Rest]) :-
    % Get player status to check for damage_bonus
    current_player(character(_, _, _, _, _, _, _, _, _, PStatus)),
    enemy_instance(EnemyId, character(EClass, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status)),
    % Calculate base damage
    roll_damage(DamageSpec, Base),
    % Calculate extra damage (if there's an extra_damage effect)
    calculate_extra_damage(Effects, ExtraDamage),
    % Calculate damage bonus buff (if player has damage_bonus status)
    ( member(damage_bonus(Bonus), PStatus)
    -> DamageBonus = Bonus
    ;  DamageBonus = 0
    ),
    TotalBase is Base + ExtraDamage + DamageBonus,
    DamageRaw is TotalBase - AC,
    ( DamageRaw =< 0 -> Damage is 0 ; Damage is DamageRaw ),
    NewHP is max(0, HP - Damage),
    % Apply status effects
    apply_all_enemies_effects(Effects, Status, NewStatus),
    retractall(enemy_instance(EnemyId, _)),
    assertz(enemy_instance(EnemyId, character(EClass, NewHP, AC, W, Inv, Skills, Act, BAct, Slots, NewStatus))),
    enemy_display_name(EnemyId, DisplayName),
    format(string(Msg), '~w takes ~w damage (after AC).', [DisplayName, Damage]),
    typewriter_line(Msg),
    apply_skill_to_all_enemies(SkillId, Name, DamageSpec, Effects, Rest).

% Use item: find corresponding effect based on item name
resolve_item_use(SelectedItemName) :-
    ( item(_Id, SelectedItemName, _Desc, heal(self, Dice, Flat))
    -> % Using item consumes 1 Bonus Action
       current_player(Char0),
       ( pay_action_cost(Char0, action_cost(0,1,0), Char1)
       -> roll_dice(Dice, DiceVal),
          Amount is DiceVal + Flat,
          Char1 = character(Class, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status),
          NewHP is HP + Amount,
          remove_one(SelectedItemName, Inv, NewInv),
          retractall(player_character(_)),
          assertz(player_character(character(Class, NewHP, AC, W, NewInv, Skills, Act, BAct, Slots, Status))),
          format(string(Msg), 'You drink ~w and recover ~w HP.', [SelectedItemName, Amount]),
          typewriter_line(Msg),
          nl
       ;  typewriter_line('Not enough Bonus Action to use an item.'),
          nl
       )
    ; item(_Id, SelectedItemName, _Desc, buff(self, damage_bonus(Bonus), duration(end_of_combat)))
    -> % Using item consumes 1 Bonus Action
       current_player(Char0),
       ( pay_action_cost(Char0, action_cost(0,1,0), Char1)
       -> Char1 = character(Class, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status),
          % Remove item from inventory
          remove_one(SelectedItemName, Inv, NewInv),
          % Add damage bonus buff to status effects
          NewStatus = [damage_bonus(Bonus)|Status],
          retractall(player_character(_)),
          assertz(player_character(character(Class, HP, AC, W, NewInv, Skills, Act, BAct, Slots, NewStatus))),
          format(string(Msg), 'You drink ~w and gain +~w damage to all attacks until the end of this combat.', [SelectedItemName, Bonus]),
          typewriter_line(Msg),
          nl
       ;  typewriter_line('Not enough Bonus Action to use an item.'),
          nl
       )
    ; nl,
      typewriter_line('(implementing...)'),
      nl
    ).

/* ======================================================================
   Combat Status Display and Skill/Item Lists
   ====================================================================== */

% Display status of all units on the battlefield
show_battle_status :-
    nl,
    typewriter_line('--- Battle Status ---'),
    current_player(character(Class, HP, AC, _, _, _, Action, BonusAction, SpellSlots, StatusEffects)),
    format(string(SP1), 'Player (~w) HP: ~w, AC: ~w, Action: ~w, Bonus Action: ~w, Spell Slots: ~w, Status: ~w', [Class, HP, AC, Action, BonusAction, SpellSlots, StatusEffects]),
    typewriter_line(SP1),
    % Dynamically display all enemies
    findall(Id, enemy_instance(Id, _), EnemyIds),
    show_all_enemies_status(EnemyIds).

show_all_enemies_status([]) :- !.
show_all_enemies_status([EnemyId|Rest]) :-
    enemy_instance(EnemyId, character(EClass, EHP, EAC, _, _, _, _, _, _, EStatus)),
    enemy_display_name(EnemyId, DisplayName),
    format(string(SE), 'Enemy ~w (~w) HP: ~w, AC: ~w, Status: ~w', [DisplayName, EClass, EHP, EAC, EStatus]),
    typewriter_line(SE),
    show_all_enemies_status(Rest).

% List current player's available skills (with numbers), and collect skill ID list
% show_available_skills(-SkillList) returns skill ID list
show_available_skills(SkillList) :-
    current_player(character(Class, _, _, _, _, _, _, _, _, _)),
    class_skill_tag(Class, Tag),
    findall(Id, (skill(Id, _, _, _, _, _, _, _, _, Tags), member(Tag, Tags), is_skill_available(Id)), SkillList),
    show_available_skills_with_numbers(SkillList, 1).

% Check if skill is available in current encounter (filter out used per_encounter(1) skills)
is_skill_available(SkillId) :-
    skill(SkillId, _, _, _, _, _, _, _, usage_limit(per_encounter(1)), _),
    !,
    \+ skill_usage_count(SkillId, used).

is_skill_available(_) :-
    true.

% Display numbered skill list
show_available_skills_with_numbers([], _).
show_available_skills_with_numbers([Id|Rest], N) :-
    skill(Id, Name, Desc, _, _, _, _, _, _, _),
    format(string(S), '~w. ~w: ~w', [N, Name, Desc]),
    typewriter_line(S),
    N1 is N + 1,
    show_available_skills_with_numbers(Rest, N1).

% List current player's inventory items (with numbers), and collect item list
% show_available_items(-ItemList) returns item list
show_available_items(ItemList) :-
    current_player(character(_, _, _, _, Inventory, _, _, _, _, _)),
    ( Inventory = []
    -> ItemList = [],
       typewriter_line('No usable items.')
    ;  ItemList = Inventory,
       show_available_items_with_numbers(ItemList, 1)
    ).

% Display numbered item list
show_available_items_with_numbers([], _).
show_available_items_with_numbers([Item|Rest], N) :-
    format(string(S), '~w. ~w', [N, Item]),
    typewriter_line(S),
    N1 is N + 1,
    show_available_items_with_numbers(Rest, N1).

/* ======================================================================
   Enemy Turn and AI System
   ====================================================================== */

% Enemy turn skeleton
enemies_turn :-
    nl,
    typewriter_line('--- Enemies Turn ---'),
    % Process enemy status effects and AI turns
    process_enemy_status_effects,
    execute_all_enemy_ai_turns,
    nl.

% Process all enemy status effects
process_enemy_status_effects :-
    findall(Id, enemy_instance(Id, _), EnemyIds),
    process_each_enemy_status(EnemyIds).

process_each_enemy_status([]) :- !.
process_each_enemy_status([EnemyId|Rest]) :-
    enemy_instance(EnemyId, character(EClass, HP, AC, W, Inv, Skills, Act, BAct, Slots, Status)),
    % Get display name
    enemy_display_name(EnemyId, DisplayName),
    % Only process status effects for alive enemies
    ( HP > 0
    -> % Process bleeding status
       ( member(bleeding, Status)
       -> NewHP is max(0, HP - 2),
          format(string(BleedMsg), '~w takes 2 damage from bleeding.', [DisplayName]),
          typewriter_line(BleedMsg)
       ;  NewHP = HP
       ),
       % Process stunned/frightened status - skip turn
       ( (member(stunned, Status) ; member(frightened, Status))
       -> format(string(StunMsg), '~w is stunned/frightened and cannot act.', [DisplayName]),
          typewriter_line(StunMsg),
          NewAct = 0
       ;  NewAct = Act
       ),
       % Process confused/shocked status - action is 0
       ( (member(confused, Status) ; member(shocked, Status))
       -> format(string(ConfMsg), '~w is confused/shocked and cannot use actions.', [DisplayName]),
          typewriter_line(ConfMsg),
          NewAct2 = 0
       ;  NewAct2 = NewAct
       ),
       % Update enemy status
       retractall(enemy_instance(EnemyId, _)),
       assertz(enemy_instance(EnemyId, character(EClass, NewHP, AC, W, Inv, Skills, NewAct2, BAct, Slots, Status)))
    ;  true  % Enemy is dead, don't process status
    ),
    process_each_enemy_status(Rest).

% Execute all enemy AI turns
execute_all_enemy_ai_turns :-
    findall(Id, enemy_instance(Id, _), EnemyIds),
    execute_each_enemy_ai_turn(EnemyIds).

execute_each_enemy_ai_turn([]) :- !.
execute_each_enemy_ai_turn([EnemyId|Rest]) :-
    enemy_instance(EnemyId, character(_, HP, _, _, _, _, Act, _, _, Status)),
    % Only enemies with HP > 0, Action > 0, and no control status can act
    ( HP > 0,
      Act > 0,
      \+ member(stunned, Status),
      \+ member(frightened, Status)
    -> ( catch(enemy_ai_turn(EnemyId),
               Error,
               (format(string(ErrorMsg), 'Error in enemy AI for ~w: ~w', [EnemyId, Error]),
                typewriter_line(ErrorMsg),
                nl))
       -> true
       ;  true
       )
    ;  true
    ),
    execute_each_enemy_ai_turn(Rest).

/* ======================================================================
   PDDL Planner Integration - Enemy AI
   ====================================================================== */

% Generate Problem.pddl file
generate_problem_pddl(EnemyId, ProblemFile) :-
    % Get enemy status
    enemy_instance(EnemyId, character(EClass, _, _, _, _, _, _, _, _, EStatus)),
    
    % Calculate status initialization strings first
    generate_enemy_status_init(EStatus, EnemyId, StatusInitStr),
    generate_blood_sacrifice_init(EClass, EnemyId, BloodSacrificeStr),
    generate_enemy_type_init(EClass, EnemyId, EnemyTypeStr),
    
    % Build Problem.pddl content
    atom_concat('(define (problem nautiloid_combat_', EnemyId, Part1),
    atom_concat(Part1, ')\n\n    (:domain nautiloid_combat)\n\n    (:objects\n        ', Part2),
    atom_concat(Part2, EnemyId, Part3),
    atom_concat(Part3, ' - enemy\n    )\n\n    (:init\n        (player_alive)\n        (enemy_alive ', Part4),
    atom_concat(Part4, EnemyId, Part5),
    atom_concat(Part5, ')\n        ', Part6),
    % Add enemy type
    atom_concat(Part6, EnemyTypeStr, Part6b),
    atom_concat(Part6b, '\n        ', Part7),
    % Add status effects
    atom_concat(Part7, StatusInitStr, Part7b),
    ( StatusInitStr = ''
    -> Part8 = Part7b
    ;  atom_concat(Part7b, '\n        ', Part8)
    ),
    atom_concat(Part8, '(enemy_has_action ', Part9),
    atom_concat(Part9, EnemyId, Part10),
    atom_concat(Part10, ')\n        (enemy_has_bonus_action ', Part11),
    atom_concat(Part11, EnemyId, Part12),
    atom_concat(Part12, ')\n        (can_act ', Part13),
    atom_concat(Part13, EnemyId, Part14),
    atom_concat(Part14, ')\n        (can_use_action ', Part15),
    atom_concat(Part15, EnemyId, Part16),
    atom_concat(Part16, ')\n        ', Part17),
    atom_concat(Part17, BloodSacrificeStr, Part18),
    ( BloodSacrificeStr = ''
    -> Part19 = Part18
    ;  atom_concat(Part18, '\n        ', Part19)
    ),
    atom_concat(Part19, '\n    )\n\n    (:goal (player_defeated))\n\n)', ProblemContent),
    
    % Write to file
    atom_string(ProblemFile, ProblemFileStr),
    open(ProblemFileStr, write, Out, [encoding(utf8)]),
    atom_string(ProblemContent, ProblemStr),
    write(Out, ProblemStr),
    close(Out).

% Generate enemy type initialization
generate_enemy_type_init(imp, EnemyId, Str) :-
    format(atom(Str), '(is_imp ~w)', [EnemyId]).
generate_enemy_type_init(cambion, EnemyId, Str) :-
    format(atom(Str), '(is_cambion ~w)', [EnemyId]).

% Generate enemy status initialization (handle temporary status)
generate_enemy_status_init(Status, EnemyId, StatusStr) :-
    findall(StatusInit, 
        (member(StatusType, Status),
         status_to_pddl_init(StatusType, EnemyId, StatusInit)
        ),
        StatusInits
    ),
    ( StatusInits = []
    -> StatusStr = ''
    ;  atomics_to_string(StatusInits, '\n        ', StatusStr)
    ).

% Convert status type to PDDL initialization
status_to_pddl_init(stunned, EnemyId, Str) :-
    format(atom(Str), '(enemy_stunned ~w)', [EnemyId]).
status_to_pddl_init(frightened, EnemyId, Str) :-
    format(atom(Str), '(enemy_frightened ~w)', [EnemyId]).
status_to_pddl_init(confused, EnemyId, Str) :-
    format(atom(Str), '(enemy_confused ~w)', [EnemyId]).
status_to_pddl_init(shocked, EnemyId, Str) :-
    format(atom(Str), '(enemy_shocked ~w)', [EnemyId]).
status_to_pddl_init(bleeding, EnemyId, Str) :-
    format(atom(Str), '(enemy_bleeding ~w)', [EnemyId]).
status_to_pddl_init(_, _, '').

% Generate Blood Sacrifice initialization (only available for Cambion)
generate_blood_sacrifice_init(cambion, EnemyId, Str) :-
    format(atom(Str), '(blood_sacrifice_available ~w)', [EnemyId]).
generate_blood_sacrifice_init(_, _, '').

% Execute enemy AI turn
enemy_ai_turn(EnemyId) :-
    enemy_instance(EnemyId, character(EClass, HP, _, _, Inv, _, _, _, _, _)),
    % Calculate max HP
    enemy_max_hp(EClass, MaxHP),
    LowHPThreshold is MaxHP * 0.4,  % Consider healing when below 40% HP
    % Priority check: does enemy need healing?
    ( HP =< LowHPThreshold,
      can_enemy_heal(EnemyId, EClass, Inv)
    -> execute_enemy_heal(EnemyId, EClass)
    ;  % Otherwise use PDDL planner to decide attack
       generate_problem_pddl(EnemyId, 'problem.pddl'),
       ( catch(
           run_pyperplan_python(path(python), 'domain.pddl', 'problem.pddl', Actions),
           Error,
           (format(user_error, 'Pyperplan error: ~w~n', [Error]), Actions = [])
         )
       -> ( Actions = []
          -> true  % No valid actions, enemy skips
          ;  execute_enemy_actions(EnemyId, Actions)
          )
       ;  true  % Planning failed, enemy skips
       )
    ).

% Check if enemy can heal
can_enemy_heal(_, imp, Inv) :-
    member('Potion of Healing', Inv), !.
can_enemy_heal(EnemyId, cambion, _) :-
    % Cambion needs to check if blood_sacrifice has been used
    \+ skill_usage_count(blood_sacrifice_used(EnemyId), used), !.

% Execute enemy healing
execute_enemy_heal(EnemyId, imp) :-
    execute_imp_use_potion(EnemyId).
execute_enemy_heal(EnemyId, cambion) :-
    execute_blood_sacrifice(EnemyId).

% Imp uses healing potion
execute_imp_use_potion(EnemyId) :-
    enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, EAct, EBAct, ESlots, EStatus)),
    ( member('Potion of Healing', EInv)
    -> % Calculate healing amount (2d4+2)
       roll_dice(dice(2, 4), DiceVal),
       HealAmount is DiceVal + 2,
       enemy_max_hp(EClass, MaxHP),
       NewEHP is min(MaxHP, EHP + HealAmount),
       % Remove potion from inventory
       remove_one('Potion of Healing', EInv, NewEInv),
       % Consume Action
       NewEAct is EAct - 1,
       % Healing clears bleeding status
       delete(EStatus, bleeding, NewEStatus),
       retractall(enemy_instance(EnemyId, _)),
       assertz(enemy_instance(EnemyId, character(EClass, NewEHP, EAC, EW, NewEInv, ESkills, NewEAct, EBAct, ESlots, NewEStatus))),
       % Display message
       enemy_display_name(EnemyId, DisplayName),
       format(string(Msg), '~w drinks a Potion of Healing and recovers ~w HP.', [DisplayName, HealAmount]),
       typewriter_line(Msg),
       nl
    ;  true  % No potion, skip
    ).

% Use python -m pyperplan to call planner
run_pyperplan_python(Python, Domain, Problem, Actions) :-
    atom_string(Problem, PStr),
    string_concat(PStr, ".soln", SolnStr),
    atom_string(SolnFile, SolnStr),
    setup_call_cleanup(
        process_create(Python, ['-m', 'pyperplan', Domain, Problem],
                       [ stdout(null),
                         stderr(pipe(Err)),
                         process(PID)
                       ]),
        read_string(Err, _, Stderr),
        close(Err)
    ),
    process_wait(PID, exit(Status)),
    ( exists_file(SolnFile)
    -> read_plan_file(SolnFile, Actions)
    ;  ( Status = exit(0)
       -> Actions = []  % Planning succeeded but no solution, return empty list
       ;  format(user_error, "Pyperplan stderr: ~s~n", [Stderr]),
          Actions = []
       )
    ).

% Execute a series of enemy actions
execute_enemy_actions(_, []) :- !.
execute_enemy_actions(EnemyId, [Action|Rest]) :-
    execute_single_enemy_action(EnemyId, Action),
    execute_enemy_actions(EnemyId, Rest).

% Execute single enemy action
execute_single_enemy_action(EnemyId, Action) :-
    functor(Action, ActionName, _),
    ( ActionName = imp_claw
    -> execute_imp_claw(EnemyId)
    ;  ActionName = everflame_slash
    -> execute_everflame_slash(EnemyId)
    ;  ActionName = blood_sacrifice
    -> execute_blood_sacrifice(EnemyId)
    ;  ActionName = clear_stunned
    -> execute_clear_status(EnemyId, stunned)
    ;  ActionName = clear_frightened
    -> execute_clear_status(EnemyId, frightened)
    ;  ActionName = clear_confused
    -> execute_clear_status(EnemyId, confused)
    ;  ActionName = clear_shocked
    -> execute_clear_status(EnemyId, shocked)
    ;  ActionName = apply_bleeding_damage
    -> execute_apply_bleeding_damage(EnemyId)
    ;  ActionName = reset_actions
    -> execute_reset_actions(EnemyId)
    ;  true  % Unknown action, ignore
    ).

% Execute Imp Claw attack
execute_imp_claw(EnemyId) :-
    enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, EAct, EBAct, ESlots, EStatus)),
    current_player(character(PClass, PHP, PAC, PW, PInv, PSkills, PAct, PBAct, PSlots, PStatus)),
    
    % Calculate damage (1d8)
    roll_dice(dice(1, 8), Damage),
    DamageAfterAC is max(0, Damage - PAC),
    NewPHP is max(0, PHP - DamageAfterAC),
    
    % Update player HP
    retractall(player_character(_)),
    assertz(player_character(character(PClass, NewPHP, PAC, PW, PInv, PSkills, PAct, PBAct, PSlots, PStatus))),
    
    % Update enemy Action
    NewEAct is EAct - 1,
    retractall(enemy_instance(EnemyId, _)),
    assertz(enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, NewEAct, EBAct, ESlots, EStatus))),
    
    % Display message
    enemy_display_name(EnemyId, DisplayName),
    format(string(Msg), '~w uses Imp Claw on you and deals ~w damage (after AC).', [DisplayName, DamageAfterAC]),
    typewriter_line(Msg),
    nl.

% Execute Cambion Everflame Slash
execute_everflame_slash(EnemyId) :-
    enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, EAct, EBAct, ESlots, EStatus)),
    current_player(character(PClass, PHP, PAC, PW, PInv, PSkills, PAct, PBAct, PSlots, PStatus)),
    
    % Calculate damage (2d8+1d2)
    roll_dice(dice(2, 8), Damage1),
    roll_dice(dice(1, 2), Damage2),
    TotalDamage is Damage1 + Damage2,
    DamageAfterAC is max(0, TotalDamage - PAC),
    NewPHP is max(0, PHP - DamageAfterAC),
    
    % Update player HP
    retractall(player_character(_)),
    assertz(player_character(character(PClass, NewPHP, PAC, PW, PInv, PSkills, PAct, PBAct, PSlots, PStatus))),
    
    % Update enemy Action
    NewEAct is EAct - 1,
    retractall(enemy_instance(EnemyId, _)),
    assertz(enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, NewEAct, EBAct, ESlots, EStatus))),
    
    % Display message
    enemy_display_name(EnemyId, DisplayName),
    format(string(Msg), '~w uses Everflame Slash on you and deals ~w damage (after AC).', [DisplayName, DamageAfterAC]),
    typewriter_line(Msg),
    nl.

% Execute Cambion Blood Sacrifice
execute_blood_sacrifice(EnemyId) :-
    % Check if already used
    ( skill_usage_count(blood_sacrifice_used(EnemyId), used)
    -> true  % Already used, skip
    ;  enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, EAct, EBAct, ESlots, EStatus)),
       % Calculate healing amount (2d10)
       roll_dice(dice(2, 10), HealAmount),
       enemy_max_hp(EClass, MaxHP),
       NewEHP is min(MaxHP, EHP + HealAmount),
       % Update enemy HP and Action
       NewEAct is EAct - 1,
       % Healing clears bleeding status
       delete(EStatus, bleeding, NewEStatus),
       retractall(enemy_instance(EnemyId, _)),
       assertz(enemy_instance(EnemyId, character(EClass, NewEHP, EAC, EW, EInv, ESkills, NewEAct, EBAct, ESlots, NewEStatus))),
       % Record skill as used
       assertz(skill_usage_count(blood_sacrifice_used(EnemyId), used)),
       % Display message
       enemy_display_name(EnemyId, DisplayName),
       format(string(Msg), '~w uses Blood Sacrifice and heals ~w HP.', [DisplayName, HealAmount]),
       typewriter_line(Msg),
       nl
    ).

% Execute clear specific status
execute_clear_status(EnemyId, StatusType) :-
    enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, EAct, EBAct, ESlots, Status)),
    
    % Remove specified status
    delete(Status, StatusType, NewStatus),
    
    retractall(enemy_instance(EnemyId, _)),
    assertz(enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, EAct, EBAct, ESlots, NewStatus))).

% Execute bleeding damage
execute_apply_bleeding_damage(EnemyId) :-
    enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, EAct, EBAct, ESlots, Status)),
    
    ( member(bleeding, Status)
    -> NewEHP is max(0, EHP - 2),
       retractall(enemy_instance(EnemyId, _)),
       assertz(enemy_instance(EnemyId, character(EClass, NewEHP, EAC, EW, EInv, ESkills, EAct, EBAct, ESlots, Status))),
       enemy_display_name(EnemyId, DisplayName),
       format(string(Msg), '~w takes 2 damage from bleeding.', [DisplayName]),
       typewriter_line(Msg),
       nl
    ;  true
    ).

% Execute reset Action
execute_reset_actions(EnemyId) :-
    enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, _, _, ESlots, Status)),
    
    retractall(enemy_instance(EnemyId, _)),
    assertz(enemy_instance(EnemyId, character(EClass, EHP, EAC, EW, EInv, ESkills, 1, 1, ESlots, Status))).

/* ======================================================================
   Post-Combat Scene - Combat Room Exploration
   ====================================================================== */

show_post_combat_scene :-
    nl,
    typewriter_line('You stand amidst the ghastly aftermath. The floor is slick with ichor, and the bodies of mind flayers, imps, and former human prisoners lie sprawled together—a grotesque tapestry of the vessel''s horrors. A crushing fatigue weighs on you, and your body cries out for a moment''s rest and a chance to search for supplies.'),
    nl,
    typewriter_line('But the chamber itself offers no respite. It is ruined and rent by the violence of the last few minutes, its organic surfaces torn and the few scattered items utterly destroyed.'),
    nl,
    typewriter_line('Only two features pierce the wreckage: A webbed structure overhead, leading to the room above. A bizarre, helix device standing sentinel nearby.'),
    nl,
    show_post_combat_options.

show_post_combat_options :-
    typewriter_line('The following objects are available for investigation:'),
    nl,
    typewriter_line('1. Webbed Structure (leads to the room above)'),
    nl,
    typewriter_line('2. Helix Device'),
    nl,
    typewriter_line('(Enter the corresponding number to investigate:)'),
    nl,
    get_post_combat_choice.

get_post_combat_choice :-
    read(Choice),
    handle_post_combat_choice(Choice).

handle_post_combat_choice(1) :-
    investigate_webbed_structure.

handle_post_combat_choice(2) :-
    investigate_helix_device.

handle_post_combat_choice(_) :-
    nl,
    typewriter_line('Invalid input. Please enter 1 or 2.'),
    nl,
    get_post_combat_choice.

/* Investigate Webbed Structure */
investigate_webbed_structure :-
    nl,
    typewriter_line('You approach the webbed structure and realize with revulsion that it is not webbing at all—it is a network of pulsating arteries, slick with viscous fluid and still very much alive. The thick vessels throb with a slow, rhythmic beat, and through their translucent walls you can glimpse unidentifiable substances being transported within.'),
    nl,
    typewriter_line('Despite its grotesque appearance, the arterial network appears remarkably sturdy. The intertwined vessels form a lattice strong enough to support your weight, offering a path upward to the chamber above.'),
    nl,
    ask_climb_webbed_structure.

ask_climb_webbed_structure :-
    typewriter_line('Do you wish to climb the arterial network to reach the next room? (y/n)'),
    nl,
    read(Answer),
    handle_climb_answer(Answer).

handle_climb_answer(y) :-
    nl,
    typewriter_line('Suppressing your disgust, you grip the slick, pulsating arteries and begin your ascent. The vessels throb beneath your fingers, warm and disturbingly alive. Hand over hand, you pull yourself upward through the organic lattice...'),
    nl,
    show_laboratory_scene.

handle_climb_answer(n) :-
    nl,
    typewriter_line('You step back from the arterial network, not yet ready to make the climb.'),
    nl,
    show_post_combat_options.

handle_climb_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_climb_webbed_structure.

/* Investigate Restoration Pod */
investigate_helix_device :-
    nl,
    typewriter_line('You approach the bizarre helix structure and realize it is a Restoration Pod—one of the Illithids'' organic healing chambers. The device still clings to a faint spark of life, its surface pulsing with a weak but steady rhythm.'),
    nl,
    typewriter_line('From within the pod, countless blue, bioluminescent tendrils extend outward, swaying gently as if sensing your presence. They appear to be some form of medical apparatus, designed to interface directly with organic tissue.'),
    nl,
    typewriter_line('One tendril reaches out and brushes against a gash on your arm. You flinch, expecting pain, but instead feel a cool, soothing sensation as the flesh knits itself together before your eyes. The wound closes, leaving only a faint scar.'),
    nl,
    typewriter_line('The pod''s interior beckons, promising complete restoration to those who dare to step within its alien embrace.'),
    nl,
    ask_enter_restoration_pod.

ask_enter_restoration_pod :-
    typewriter_line('Do you wish to step into the Restoration Pod and receive its full healing? (y/n)'),
    nl,
    read(Answer),
    handle_restoration_pod_answer(Answer).

handle_restoration_pod_answer(y) :-
    nl,
    typewriter_line('You steel your nerves and step into the Restoration Pod. The blue tendrils immediately envelop you, their cool touch spreading across your entire body. For a moment, panic grips you—but then a wave of profound relief washes over your battered form.'),
    nl,
    typewriter_line('The tendrils work with alien precision, mending torn flesh, soothing bruised muscles, and easing the deep ache in your bones. When they finally release you, you feel completely restored, as if the horrors of the past hours had never touched your body.'),
    nl,
    % Restore player HP to maximum
    restore_player_to_full_hp,
    typewriter_line('(Your HP has been fully restored.)'),
    nl, nl,
    show_post_combat_options.

handle_restoration_pod_answer(n) :-
    nl,
    typewriter_line('You step back from the Restoration Pod, unwilling to trust the alien device completely. Perhaps caution is wise in this place.'),
    nl, nl,
    show_post_combat_options.

handle_restoration_pod_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_enter_restoration_pod.

% Fully restore player status
restore_player_to_full_hp :-
    current_player(character(Class, _, AC, W, Inv, Skills, _, _, _, _)),
    class_max_hp(Class, MaxHP),
    class_max_spell_slots(Class, MaxSpellSlots),
    retractall(player_character(_)),
    % Restore HP to max, reset Action and BonusAction to 1, restore spell slots, clear all status effects
    assertz(player_character(character(Class, MaxHP, AC, W, Inv, Skills, 1, 1, MaxSpellSlots, []))),
    % Reset all skill usage counts (per_encounter skills can be used again)
    retractall(skill_usage_count(_, _)).

/* ======================================================================
   Laboratory Scene
   ====================================================================== */

show_laboratory_scene :-
    nl,
    typewriter_line('You haul yourself up through the arterial network and emerge into a new chamber. This appears to be some manner of laboratory—a place of unspeakable experimentation.'),
    nl,
    typewriter_line('Dominating the center of the room is a massive cylindrical apparatus. At its apex sits an enormous sea urchin-like organism, its spines twitching with malevolent life. You can see it actively drawing some form of nutrient from below, the sustenance traveling through a network of tubes that extend outward to several restraining chairs along the walls.'),
    nl,
    typewriter_line('The prisoners strapped to these chairs are beyond saving—their bodies withered and desiccated, drained of all vitality to feed the grotesque creature above. Behind the chairs stand rows of Mind Flayer Pods, each accompanied by a fleshy control console.'),
    nl,
    typewriter_line('Suddenly, a sharp tapping sound breaks the silence. Your eyes dart to its source: one of the pods. Through the murky glass, you see a half-elf woman in clerical vestments, desperately striking the interior of her prison. A survivor! Finally, a living soul in this nightmare.'),
    nl,
    show_laboratory_options.

show_laboratory_options :-
    typewriter_line('The following objects are available for investigation:'),
    nl,
    typewriter_line('1. Mind Flayer Pod (containing the half-elf cleric)'),
    nl,
    typewriter_line('2. Fleshy Control Console (next to the pod)'),
    nl,
    typewriter_line('3. Passage (sign reads: "Laboratory 2")'),
    nl,
    typewriter_line('4. Passage (sign reads: "Helm")'),
    nl,
    typewriter_line('(Enter the corresponding number to investigate:)'),
    nl,
    get_laboratory_choice.

get_laboratory_choice :-
    read(Choice),   
    handle_laboratory_choice(Choice).

handle_laboratory_choice(1) :-
    investigate_cleric_pod.

handle_laboratory_choice(2) :-
    investigate_control_console.

handle_laboratory_choice(3) :-
    investigate_laboratory2_passage.

handle_laboratory_choice(4) :-
    investigate_helm_passage.

handle_laboratory_choice(_) :-
    nl,
    typewriter_line('Invalid input. Please enter 1, 2, 3, or 4.'),
    nl,
    get_laboratory_choice.

/* Investigate the Mind Flayer Pod containing the cleric */
investigate_cleric_pod :-
    ( cleric_rescued(true)
    -> nl,
       typewriter_line('The pod stands empty now, its glass panel hanging open. Residual amniotic fluid drips slowly from its interior. A few motes of golden light still drift lazily in the air—the only remaining trace of the Aasimar who ascended to the heavens.'),
       nl, nl,
       show_laboratory_options
    ;  nl,
       typewriter_line('You approach the pod and peer through the thick, clouded glass. The half-elf woman inside appears young, her dark hair matted with the pod''s viscous fluid. Despite her imprisonment, her eyes burn with fierce determination.'),
       nl,
       typewriter_line('She presses her palm against the glass, her lips moving urgently. Though the pod muffles her voice, you can make out fragments: "...the console... release mechanism... please..."'),
       nl,
       typewriter_line('A holy symbol of Selûne hangs around her neck—she is indeed a cleric, a servant of the Moonmaiden. Her presence here might prove invaluable... if you can find a way to free her.'),
       nl, nl,
       show_laboratory_options
    ).

/* Investigate control console */
investigate_control_console :-
    ( cleric_rescued(true)
    -> nl,
       typewriter_line('The control console is now dormant, the Mysterious Slate still embedded in its slot. The empty pod beside it stands open, and faint motes of golden light drift through the air where the Aasimar once stood.'),
       nl, nl,
       show_laboratory_options
    ;  nl,
       typewriter_line('You examine the control console beside the cleric''s pod. It is an intricate device of alien design, and you have no idea how to operate it.'),
       nl,
       typewriter_line('You reach out toward the console. It quickly senses your presence and begins to writhe slowly, its organic surface undulating with apparent recognition. But after a moment, it falls still again—like an automated response that failed to authenticate.'),
       nl,
       typewriter_line('You notice a narrow, bar-shaped slot on the console''s surface. Perhaps some kind of key is required.'),
       nl,
       ( has_slate(true)
       -> typewriter_line('The slot appears to be perfectly shaped for the Mysterious Slate you carry. Do you wish to insert it? (y/n)'),
          nl,
          read(Answer),
          handle_insert_slate_answer(Answer)
       ;  nl,
          show_laboratory_options
       )
    ).

handle_insert_slate_answer(y) :-
    nl,
    typewriter_line('You slide the Mysterious Slate into the slot. It fits perfectly, the glyphs on its surface flaring to life as they make contact with the console.'),
    nl,
    typewriter_line('The console shudders and emits a low, resonant hum. The pod beside you responds immediately—its seal hisses open, releasing a cascade of amniotic fluid.'),
    nl,
    typewriter_line('The half-elf woman stumbles out, gasping for fresh air. She steadies herself against you, her eyes wide with relief and gratitude.'),
    nl,
    typewriter_line('But as she rises to her full height, something extraordinary happens. A soft, golden light begins to emanate from within her, growing brighter with each passing moment. Her features shift subtly—her skin takes on an ethereal luminescence, and for a brief instant, you glimpse the faint outline of radiant wings behind her shoulders.'),
    nl,
    typewriter_line('"I am not merely a cleric of Selûne," she speaks, her voice now carrying an otherworldly resonance. "I am an Aasimar—a child of the divine, touched by celestial blood. The Mind Flayers sought to corrupt my essence for their foul experiments."'),
    nl,
    typewriter_line('She regards you with eyes that seem to hold the light of distant stars. "You have freed me from a fate worse than death, mortal. For this, I offer you a gift—a portion of my divine power. I can awaken the spark of the sacred within you, transforming you into a Paladin, a holy warrior blessed by the gods."'),
    nl,
    ask_accept_divine_gift.

ask_accept_divine_gift :-
    typewriter_line('Do you accept the Aasimar''s divine gift and become a Paladin? (y/n)'),
    nl,
    read(Answer),
    handle_divine_gift_answer(Answer).

handle_divine_gift_answer(y) :-
    nl,
    typewriter_line('You kneel before the Aasimar, accepting her blessing.'),
    nl,
    typewriter_line('She raises her hands and begins to chant in a language older than mortal tongues—words of power that resonate with the very fabric of creation:'),
    nl,
    typewriter_line('"Lux aeterna, descende! Per gratiam caelorum, hunc servum tuum consecro. Fiat miles sanctus, defensor innocentium, malleus maleficorum!"'),
    nl,
    typewriter_line('A blinding column of holy light pierces through the organic ceiling of the Nautiloid, striking you with divine force. But instead of pain, you feel only warmth—a profound sense of purpose flooding through your being. Your wounds heal, your spirit strengthens, and sacred power courses through your veins.'),
    nl,
    typewriter_line('When the light fades, you rise as something more than you were. You are now a Paladin.'),
    nl,
    % Transform to Paladin
    transform_to_paladin,
    current_player(NewChar),
    display_character_info(NewChar),
    nl,
    typewriter_line('The Aasimar smiles, her form beginning to shimmer. "Use this power wisely, champion. The darkness you face is vast, but the light within you is now eternal."'),
    nl,
    typewriter_line('A pillar of radiant light envelops her, and with a final, serene smile, she ascends—vanishing into the heavens, leaving only motes of golden light drifting in her wake.'),
    nl,
    retractall(cleric_rescued(_)),
    assertz(cleric_rescued(true)),
    nl,
    show_laboratory_options.

handle_divine_gift_answer(n) :-
    nl,
    typewriter_line('You shake your head respectfully. "I am grateful for the offer, but I must walk my own path."'),
    nl,
    typewriter_line('The Aasimar''s expression softens with admiration. "Such strength of will, such humility. You saved me expecting nothing in return—a rare virtue in these dark times. Perhaps you are already more than you know."'),
    nl,
    typewriter_line('She places a gentle hand upon your shoulder. "May the light guide your steps, brave soul. We shall not meet again in this realm, but know that the gods have taken notice of your deeds this day."'),
    nl,
    typewriter_line('A pillar of radiant light envelops her, and with a final, serene smile, she ascends—vanishing into the heavens, leaving only motes of golden light drifting in her wake.'),
    nl,
    retractall(cleric_rescued(_)),
    assertz(cleric_rescued(true)),
    nl,
    show_laboratory_options.

handle_divine_gift_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_accept_divine_gift.

% Transform to Paladin
transform_to_paladin :-
    paladin(NewChar),
    retractall(player_character(_)),
    assertz(player_character(NewChar)),
    % Reset skill usage counts
    retractall(skill_usage_count(_, _)).

handle_insert_slate_answer(n) :-
    nl,
    typewriter_line('You decide not to insert the Slate just yet.'),
    nl, nl,
    show_laboratory_options.

handle_insert_slate_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    typewriter_line('Do you wish to insert the Slate? (y/n)'),
    nl,
    read(Answer),
    handle_insert_slate_answer(Answer).

/* Investigate passage to Laboratory 2 */
investigate_laboratory2_passage :-
    nl,
    typewriter_line('You approach the passage marked "Laboratory 2." The corridor beyond is dimly lit by bioluminescent growths, its walls lined with more of the ship''s organic architecture.'),
    nl,
    ask_enter_laboratory2.

ask_enter_laboratory2 :-
    typewriter_line('Do you wish to proceed to Laboratory 2? (y/n)'),
    nl,
    read(Answer),
    handle_enter_laboratory2_answer(Answer).

handle_enter_laboratory2_answer(y) :-
    nl,
    typewriter_line('You step through the passage into Laboratory 2...'),
    nl,
    show_laboratory2_scene.

handle_enter_laboratory2_answer(n) :-
    nl,
    typewriter_line('You decide to stay in the current laboratory for now.'),
    nl, nl,
    show_laboratory_options.

handle_enter_laboratory2_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_enter_laboratory2.

/* ======================================================================
   Laboratory 2 Scene
   ====================================================================== */

show_laboratory2_scene :-
    nl,
    typewriter_line('You enter Laboratory 2. The chamber is filled with more Mind Flayer Pods and their accompanying control consoles, arranged in neat rows like some grotesque nursery. Most of the pods remain sealed, their occupants—if any—hidden behind clouded glass.'),
    nl,
    typewriter_line('However, one pod stands open, its glass panel retracted. Beside it, sprawled on the fleshy floor, lies a corpse. The body appears fresh—death came recently to this unfortunate soul. Whatever killed them, they managed to escape their pod first.'),
    nl,
    show_laboratory2_options.

show_laboratory2_options :-
    typewriter_line('The following objects are available for investigation:'),
    nl,
    typewriter_line('1. Passage (return to Laboratory 1)'),
    nl,
    typewriter_line('2. Corpse'),
    nl,
    typewriter_line('(Enter the corresponding number to investigate:)'),
    nl,
    get_laboratory2_choice.

get_laboratory2_choice :-
    read(Choice),
    handle_laboratory2_choice(Choice).

handle_laboratory2_choice(1) :-
    nl,
    typewriter_line('You return through the passage to Laboratory 1.'),
    nl,
    show_laboratory_options.

handle_laboratory2_choice(2) :-
    investigate_corpse.

handle_laboratory2_choice(_) :-
    nl,
    typewriter_line('Invalid input. Please enter 1 or 2.'),
    nl,
    get_laboratory2_choice.

/* Investigate corpse */
investigate_corpse :-
    nl,
    typewriter_line('You kneel beside the corpse and examine it closely. The body belongs to a human male, dressed in the tattered remnants of what might once have been scholarly robes. His face is frozen in an expression of terror, eyes wide and mouth agape.'),
    nl,
    typewriter_line('There are no visible wounds on the body—whatever killed him left no external marks. Perhaps the shock of the tadpole insertion, or some other Illithid horror, stopped his heart.'),
    nl,
    ( has_slate(true)
    -> typewriter_line('You have already searched this body and found the Mysterious Slate.'),
       nl, nl,
       show_laboratory2_options
    ;  typewriter_line('As you search the body, your fingers close around something hard tucked into an inner pocket. You withdraw a dark stone tablet—a Mysterious Slate, its surface etched with glowing glyphs that pulse faintly at your touch.'),
       nl,
       typewriter_line('The glyphs resonate with the parasite in your skull, sending a shiver down your spine. This artifact clearly holds significance to the Mind Flayers.'),
       nl,
       typewriter_line('(You have obtained the Mysterious Slate.)'),
       retractall(has_slate(_)),
       assertz(has_slate(true)),
       nl, nl,
       show_laboratory2_options
    ).

/* ======================================================================
   Passage to the Helm
   ====================================================================== */

/* Investigate passage to the Helm */
investigate_helm_passage :-
    nl,
    typewriter_line('You examine the passage leading to the Helm. This corridor appears more heavily reinforced than the others, its walls thick with armored chitin. A faint, pulsing glow emanates from somewhere ahead.'),
    nl,
    typewriter_line('The Helm—the nerve center of the Nautiloid. If there is any way to control this vessel or find an escape route, it would be there. But such a critical location is unlikely to be unguarded.'),
    nl,
    ask_enter_helm.

ask_enter_helm :-
    typewriter_line('Do you wish to proceed to the Helm? (y/n)'),
    nl,
    read(Answer),
    handle_enter_helm_answer(Answer).

handle_enter_helm_answer(y) :-
    nl,
    typewriter_line('You steel yourself and step through the passage toward the Helm...'),
    nl,
    show_helm_scene.

handle_enter_helm_answer(n) :-
    nl,
    typewriter_line('You decide to stay in the laboratory for now.'),
    nl, nl,
    show_laboratory_options.

handle_enter_helm_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_enter_helm.

/* ======================================================================
   Helm Scene - Boss Battle
   ====================================================================== */

show_helm_scene :-
    nl,
    typewriter_line('You emerge into the Helm—the nerve center of the Nautiloid. The chamber is vast and circular, dominated by a massive organic control terminal at its far end. Bioluminescent veins pulse along the walls, casting an eerie glow over the scene.'),
    nl,
    typewriter_line('Before you stands a towering Cambion—a fiend of terrible power, its skin the color of dried blood, great bat-like wings folded against its back. In its clawed hand, it grips a blade wreathed in eternal flame: the Everburn Blade.'),
    nl,
    typewriter_line('The Cambion is poised over the last surviving Mind Flayer, a pitiful creature crawling desperately toward the control terminal. The Illithid''s tentacles twitch weakly as it drags itself across the floor, still clinging to some futile hope of regaining control of its vessel.'),
    nl,
    typewriter_line('But hope dies swiftly here. With a contemptuous snarl, the Cambion brings down its flaming blade. The Mind Flayer''s squid-like head separates from its body and rolls across the floor, tentacles spasming twice before falling still.'),
    nl,
    show_helm_class_specific_scene.

show_helm_class_specific_scene :-
    current_player(character(ranger, _, _, _, _, _, _, _, _, _)),
    !,
    nl,
    typewriter_line('The Cambion stands over its kill, savoring its victory. It lifts the severed head, examining it with cruel satisfaction, entirely unaware of your presence in the shadows.'),
    nl,
    typewriter_line('Your Ranger instincts scream opportunity. The fiend is distracted, vulnerable. You can strike from the shadows and seize the advantage of surprise!'),
    nl,
    setup_boss_encounter,
    start_combat_loop.

show_helm_class_specific_scene :-
    nl,
    typewriter_line('The Cambion kicks the Mind Flayer''s corpse aside and turns—its burning eyes immediately locking onto you. A cruel smile spreads across its demonic features.'),
    nl,
    typewriter_line('"Another mortal seeks to challenge Zhalk?" it growls, raising the Everburn Blade. "Come then, worm. Your soul will make a fine trophy."'),
    nl,
    typewriter_line('Two Imps flutter down from the shadows above, taking positions beside their master. The battle for the Helm begins!'),
    nl,
    setup_boss_encounter,
    start_combat_loop.

/* ======================================================================
   Boss Victory Scene and Game Ending
   ====================================================================== */

show_boss_victory_scene :-
    nl,
    typewriter_line('You rush to the control terminal at the center of the Helm. The organic interface pulses with alien life, its surface covered in writhing tentacles and glowing glyphs.'),
    nl,
    typewriter_line('Examining the device, you realize it is some manner of planar transportation system. A single tentacle extends from beneath the console, inscribed with the word "Nautiloid"—the name of this vessel.'),
    nl,
    typewriter_line('From the ceiling above, dozens more tentacles hang down, each bearing a different inscription: "Feywild," "Shadowfell," "Material Plane," "Elysium," and many others you do not recognize. All are names of planes—dimensions of existence beyond your own.'),
    nl,
    typewriter_line('The mechanism becomes clear: by connecting the ship''s tentacle to one of the destination tentacles, you can transport the Nautiloid to that plane.'),
    nl,
    typewriter_line('Without hesitation, you grasp the "Nautiloid" tentacle and connect it to "Material Plane." The organic cables fuse together with a wet, squelching sound. The console thrums with power.'),
    nl,
    typewriter_line('One small movement is all it takes now. One twitch of your hand, and you will be home.'),
    nl,
    ask_activate_portal.

ask_activate_portal :-
    typewriter_line('Do you wish to activate the planar transport and return to the Material Plane? (y/n)'),
    nl,
    read(Answer),
    handle_activate_portal_answer(Answer).

handle_activate_portal_answer(y) :-
    nl,
    typewriter_line('You activate the transport.'),
    nl,
    typewriter_line('Reality tears apart around you. The Nautiloid shudders violently as it is ripped from Avernus and hurled across the planes. Through the organic viewports, you glimpse impossible vistas—swirling chaos, infinite darkness, blinding light—as the vessel tumbles through the space between worlds.'),
    nl,
    typewriter_line('Then, with a thunderous crash, the Nautiloid breaches into the Material Plane. But the journey has taken its toll. The ship groans, its organic systems failing, its hull rupturing. You are thrown from your feet as the vessel plummets toward the ground below.'),
    nl,
    typewriter_line('The last thing you see before impact is a vast, green forest rushing up to meet you...'),
    nl, nl,
    typewriter_line('========================================'),
    typewriter_line('           TO BE CONTINUED...'),
    typewriter_line('========================================'),
    nl,
    typewriter_line('Congratulations! You have escaped the Nautiloid and completed the prologue of your adventure.'),
    nl,
    typewriter_line('Thank you for playing "Escape from Nautiloid"!'),
    nl.

handle_activate_portal_answer(n) :-
    nl,
    typewriter_line('You hesitate, your hand hovering over the connection. Perhaps there is more to explore, more to discover aboard this vessel...'),
    nl,
    typewriter_line('But deep down, you know the truth. There is nothing left here but death and madness. The Material Plane awaits.'),
    nl,
    ask_activate_portal.

handle_activate_portal_answer(_) :-
    nl,
    typewriter_line('Please answer with y or n.'),
    nl,
    ask_activate_portal.
