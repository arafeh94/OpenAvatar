const IdGenerator = (prefix = '') => {
    return () => {
        const timestamp = Date.now().toString(36);
        const randomPart = Math.random().toString(36).substring(2, 10);
        return `${prefix}${timestamp}-${randomPart}`;
    };
};

class Tools {
    static isDict(item) {
        return typeof item === 'object' && item !== null && !Array.isArray(item);
    }
}

class Fetcher {
    constructor(base_url = null) {
        this.base_url = base_url;
    }

    async fetch(route, get = {}, post = null) {
        const url = new URL(`${this.base_url}/${route}`);

        Object.keys(get).forEach(key => url.searchParams.append(key, get[key]));

        const options = {
            method: post ? 'POST' : 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (post) {
            options.body = JSON.stringify(post);
        }

        return fetch(url, options)
    }
}

class AvatarRTCClient extends Fetcher {
    static OPEN = 'onopen';
    static CLOSE = 'onclose';
    static MESSAGE = 'onmessage';
    static TRACK = 'ontrack';

    constructor(url, video_element, audio_element) {
        super(url);
        this.video = video_element;
        this.audio = audio_element;

        this.peer = new RTCPeerConnection();
        this.token = null;
        this.events = {};
        this.channel = null;
        this.peer.ondatachannel = (event) => {
            const channel = event.channel;
            channel.onopen = this.onopen.bind(this);
            channel.onclose = this.onclose.bind(this);
            channel.onmessage = this.onmessage.bind(this);
            this.channel = channel;
        }
        this.peer.ontrack = (event) => {
            this.ontrack(event)
        }
        this.callbacks = {}
        this.idGenerator = IdGenerator();
    }

    async register(persona) {
        const response = await this.fetch('register', {'persona': persona})
            .then(response => response.json());
        this.token = response.token;
        this.idGenerator = IdGenerator(this.token + '-');
        await this.peer.setRemoteDescription(response.sdp);
        await this.peer.setLocalDescription(await this.peer.createAnswer());
        const params = {'token': this.token, 'sdp': JSON.stringify(this.peer.localDescription)}
        const confirm = await this.fetch('confirm', params).then(response => response.json());
        return confirm.status === 'accepted';
    }

    onopen() {
        this.events[AvatarRTCClient.OPEN]?.();
    }

    onclose() {
        this.events[AvatarRTCClient.CLOSE]?.();
    }

    onmessage(event) {
        this.events[AvatarRTCClient.MESSAGE]?.(event.data);
        try {
            const js_packet = JSON.parse(event.data);
            this.fire_callbacks(js_packet);
            js_packet.status === 'ended' && delete this.callbacks[js_packet.id];
        } catch (err) {
            console.log('Unrefined packet received: ' + event.data, err);
        }
    }

    fire_callbacks(js_packet) {
        const _callback = this.callbacks[js_packet.id];
        if (Tools.isDict(_callback)) {
            _callback[js_packet.status] && _callback[js_packet.status](js_packet.payload);
        } else if (typeof _callback === 'function') {
            _callback && _callback(js_packet.payload);
        }
    }

    ontrack(event) {
        this.events[AvatarRTCClient.TRACK]?.(event);
        const [stream] = event.streams;
        if (event.track.kind === 'video') {
            this.video.srcObject = stream;
        } else if (event.track.kind === 'audio') {
            this.audio.srcObject = stream;
        }
    }

    on(event, callback) {
        this.events[event] = callback;
    }

    request(packet, callbacks) {
        const _id = this.idGenerator()
        packet = {'payload': packet, 'id': _id, 'type': 'json'};
        this.channel.send(JSON.stringify(packet));
        this.callbacks[_id] = callbacks;
    }
}