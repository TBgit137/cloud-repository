(define (problem nautiloid_combat_cambion1)

    (:domain nautiloid_combat)

    (:objects
        cambion1 - enemy
    )

    (:init
        (player_alive)
        (enemy_alive cambion1)
        (is_cambion cambion1)
        (enemy_has_action cambion1)
        (enemy_has_bonus_action cambion1)
        (can_act cambion1)
        (can_use_action cambion1)
        (blood_sacrifice_available cambion1)
        
    )

    (:goal (player_defeated))

)