(define (domain nautiloid_combat)

    (:requirements :equality :negative-preconditions :typing :adl)

    (:types
        enemy - object
    )

    (:predicates
        ; Life status
        (player_alive)
        (enemy_alive ?enemy - enemy)

        ; Enemy type
        (is_imp ?enemy - enemy)
        (is_cambion ?enemy - enemy)

        ; Enemy temporary status effects (cleared after one round)
        (enemy_stunned ?enemy - enemy)
        (enemy_frightened ?enemy - enemy)
        (enemy_confused ?enemy - enemy)
        (enemy_shocked ?enemy - enemy)

        ; Enemy persistent status effects
        (enemy_bleeding ?enemy - enemy)

        ; Enemy action abilities
        (can_act ?enemy - enemy)
        (can_use_action ?enemy - enemy)
        (enemy_has_action ?enemy - enemy)
        (enemy_has_bonus_action ?enemy - enemy)

        ; Skill usage limits
        (blood_sacrifice_available ?enemy - enemy)

        ; Goal
        (player_defeated)
    )

    ; Imp attacks player with claw (only for imps)
    (:action imp_claw
        :parameters (?enemy - enemy)
        :precondition (and
            (is_imp ?enemy)
            (can_act ?enemy)
            (can_use_action ?enemy)
            (enemy_has_action ?enemy)
            (player_alive)
        )
        :effect (and
            (not (enemy_has_action ?enemy))
            (player_defeated)
        )
    )

    ; Cambion attacks player with everflame slash (only for cambions)
    (:action everflame_slash
        :parameters (?enemy - enemy)
        :precondition (and
            (is_cambion ?enemy)
            (can_act ?enemy)
            (can_use_action ?enemy)
            (enemy_has_action ?enemy)
            (player_alive)
        )
        :effect (and
            (not (enemy_has_action ?enemy))
            (player_defeated)
        )
    )

    ; Cambion uses blood sacrifice to heal itself (only for cambions)
    (:action blood_sacrifice
        :parameters (?enemy - enemy)
        :precondition (and
            (is_cambion ?enemy)
            (can_act ?enemy)
            (enemy_has_action ?enemy)
            (blood_sacrifice_available ?enemy)
        )
        :effect (and
            (not (enemy_has_action ?enemy))
            (not (blood_sacrifice_available ?enemy))
        )
    )

    ; Clear stunned status
    (:action clear_stunned
        :parameters (?enemy - enemy)
        :precondition (and
            (enemy_stunned ?enemy)
        )
        :effect (and
            (not (enemy_stunned ?enemy))
            (can_act ?enemy)
            (can_use_action ?enemy)
        )
    )

    ; Clear frightened status
    (:action clear_frightened
        :parameters (?enemy - enemy)
        :precondition (and
            (enemy_frightened ?enemy)
        )
        :effect (and
            (not (enemy_frightened ?enemy))
            (can_act ?enemy)
            (can_use_action ?enemy)
        )
    )

    ; Clear confused status
    (:action clear_confused
        :parameters (?enemy - enemy)
        :precondition (and
            (enemy_confused ?enemy)
        )
        :effect (and
            (not (enemy_confused ?enemy))
            (can_act ?enemy)
            (can_use_action ?enemy)
        )
    )

    ; Clear shocked status
    (:action clear_shocked
        :parameters (?enemy - enemy)
        :precondition (and
            (enemy_shocked ?enemy)
        )
        :effect (and
            (not (enemy_shocked ?enemy))
            (can_act ?enemy)
            (can_use_action ?enemy)
        )
    )

    ; Apply bleeding damage at the start of round
    (:action apply_bleeding_damage
        :parameters (?enemy - enemy)
        :precondition (and
            (enemy_bleeding ?enemy)
        )
        :effect (and
            (enemy_bleeding ?enemy)
        )
    )

    ; Reset actions at the start of round
    (:action reset_actions
        :parameters (?enemy - enemy)
        :precondition (and
            (enemy_alive ?enemy)
        )
        :effect (and
            (enemy_has_action ?enemy)
            (enemy_has_bonus_action ?enemy)
        )
    )

)
