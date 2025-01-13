class AvatarClient {
    static template = `
            <video id="avatar_video_1" preload="auto" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
                Your browser does not support the video tag.
            </video>

            <video id="avatar_video_2" preload="auto" style="visibility: hidden; position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
                Your browser does not support the video tag.
            </video>
        `;

    constructor(containerId, host, persona = "lisa_casual_720_pl") {
        this.#create(containerId)
        this.host = host
        this.persona = persona
        this.video1 = document.getElementById('avatar_video_1');
        this.video2 = document.getElementById('avatar_video_2');
        this.players = [this.video1, this.video2]
        this.currentPlayer = null
        this.token = null
        this.players.forEach((player) => {
            player.onerror = (event) => {
                this.other(event.target).onended = () => {
                    this.stream(null, true)
                }
            }
        })
    }

    #create(containerId) {
        if (!document.getElementById("avatar_video_1") &&
            !document.getElementById("avatar_video_2")) {
            const container = document.getElementById(containerId);
            container.innerHTML = AvatarClient.template;
        }
    }

    getNextUrl() {
        const baseUrl = this.host + "/stream_next"
        return `${baseUrl}?token=${this.token}&persona=${this.persona}&t=${new Date().getTime()}`;
    }

    getIdleUrl() {
        return this.host + `/idle?persona=${this.persona}&t=1`
    }

    isPlaying(player) {
        return player.currentTime > 0 && !player.paused && !player.ended
            && player.readyState > player.HAVE_CURRENT_DATA;
    }

    load(url, player, onended = null, oncanplay = null, wait = true) {
        const other = this.other(player);
        player.src = url
        player.load()
        player.onended = onended
        player.oncanplaythrough = () => {
            if (this.isPlaying(other) && wait) {
                other.onended = () => {
                    oncanplay && oncanplay(player);
                    this.play(player)
                }
            } else {
                oncanplay && oncanplay(player);
                this.play(player)
            }
        }
    }

    other(player) {
        return (player === this.video2) ? this.video1 : this.video2;
    }

    play(player) {
        const other = this.other(player)
        other.style.visibility = 'hidden'
        other.pause()

        this.currentPlayer = player
        player.style.visibility = 'visible'
        player.play()
    }

    nextPlayer() {
        if (this.currentPlayer) {
            return this.other(this.currentPlayer)
        }
        return this.video1
    }

    mute(val) {
        this.players.forEach((player) => {
            player.muted = val
        })
    }

    stream(token, wait) {
        this.token = token
        const url = this.token ? this.getNextUrl() : this.getIdleUrl()
        onended = (element) => {
            const other = this.other(element.target)
            this.play(other)
        }
        oncanplay = (player) => {
            const other = this.other(player)
            this.load(this.getNextUrl(), other, onended, oncanplay)
        }
        this.load(url, this.nextPlayer(), onended, oncanplay, wait)
    }

    measureLatency(listener) {
        this.latency = 0;
        const start = performance.now();
        this.nextPlayer().onplaying = () => {
            if (this.latency !== 0) return;
            const end = performance.now();
            this.latency = (end - start) / 1000;
            console.log(this.latency);
            if (listener) {
                listener(this.latency);
            }
        }
    }
}

