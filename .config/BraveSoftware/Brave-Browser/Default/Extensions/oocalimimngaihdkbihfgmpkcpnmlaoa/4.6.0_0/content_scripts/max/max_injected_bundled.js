/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
var __webpack_exports__ = {};

if (!window.videoIdScriptLoaded) {
    console.log("videoID Script Loaded");
    window.videoIdScriptLoaded = true;
    window.addEventListener('MaxVideoMessage', function (event) {
        if (event.detail) {
            var type = event.detail.type;
            if (type === 'getAdList') {
                let adList = getAllAds();
                if (adList.length === 0) {
                    const tmpAdList = document.querySelector('[data-testid="player-ux-scrubber-track"]').getAttribute('data-ad-slots');
                    if (tmpAdList) {
                        adList = JSON.parse(tmpAdList);
                    }
                }
                console.log("AD LIST IS", adList);
                const titleEvent = new CustomEvent('FromNode', { detail: { type: 'getAd', adList } });
                window.dispatchEvent(titleEvent);
            }
        }
    });
}
const findReactProps = function () {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m;
    const elementRoot = document.querySelector('#overlay-root');
    if (elementRoot == null) {
        return null;
    }
    const keys = Object.keys(elementRoot);
    let key = null;
    for (let i = 0; i < keys.length; i++) {
        if (keys[i].startsWith('__reactFiber')) {
            key = keys[i];
            break;
        }
    }
    if (key == null) {
        return null;
    }
    if (typeof elementRoot[key] === 'undefined') {
        return null;
    }
    return (_m = (_l = (_k = (_j = (_h = (_g = (_f = (_e = (_d = (_c = (_b = (_a = elementRoot[key].return.pendingProps) === null || _a === void 0 ? void 0 : _a.playerCallbacks) === null || _b === void 0 ? void 0 : _b.mediator) === null || _c === void 0 ? void 0 : _c._playerControls) === null || _d === void 0 ? void 0 : _d._playbackEngineAdapter) === null || _e === void 0 ? void 0 : _e._playbackEngine) === null || _f === void 0 ? void 0 : _f._player) === null || _g === void 0 ? void 0 : _g._playerStateTracker) === null || _h === void 0 ? void 0 : _h._playerState) === null || _j === void 0 ? void 0 : _j.variant) === null || _k === void 0 ? void 0 : _k.video) === null || _l === void 0 ? void 0 : _l._periodMap) === null || _m === void 0 ? void 0 : _m._storage;
};
const getAllAds = () => {
    try {
        let adElem = findReactProps();
        let positions = [];
        if (!adElem) {
            return [];
        }
        adElem === null || adElem === void 0 ? void 0 : adElem.forEach((a, key) => {
            positions.push([a.start * 1000, a.end * 1000]);
            // a.video.forEach((b) => {
            //   let segm = b.getCachedDetails()?.segments;
            //   if (segm) {
            //     positions.push([segm[0].start * 1000, segm[segm.length - 1].end * 1000]);
            //   }
            // });
        });
        // Remove duplicates
        let uniquePositions = Array.from(new Set(positions.map(JSON.stringify)), JSON.parse);
        // Convert to an array of objects with startTime and duration properties
        let finalPositions = uniquePositions.map((position) => ({
            adStartTime: position[0],
            adDuration: position[1] - position[0]
        }));
        let filteredPositions = finalPositions.filter((position) => position.adDuration <= (61000));
        return filteredPositions;
    }
    catch (err) {
        console.error("Error is", err);
        return [];
    }
};

/******/ })()
;