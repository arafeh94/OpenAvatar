function onBeginAction() {
    if (!vars.done2 && skill.level == 2 && vars.engaged && !skill.unit.isEngaged()) {
        if (skill.unit.addStatus(Status.Inspiration, vars.value1, true)) {
            spawnFx();
            vars.done2 = true;
        }
    }
    if (!skill.unit.isEngaged()) {
        vars.engaged = false;
    }
}

function
onSkillPlayed(s) {
    if (!vars.done1 && !vars.engaged && skill.unit.isEngaged()) {
        vars.engaged = true;
        var
            b1 = skill.unit.addStatus(Status.Riposte);
        var
            b2 = skill.unit.addStatus(Status.Fury);
        if (b1 || b2) {
            spawnFx();
            vars.done1 = true;
        }
    }
}

function
onEndTurn() {
    vars.done1 = false;
    vars.done2 = false;
}
