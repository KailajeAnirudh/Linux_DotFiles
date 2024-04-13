/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
var __webpack_exports__ = {};

;// CONCATENATED MODULE: ./src/Teleparty/BrowseScripts/NativePartyHandler.ts
const addNativePartyHandler = (tryAddButton) => {
    setInterval(() => {
        try {
            const buttons = tryAddButton();
            if (buttons) {
                for (const button of buttons) {
                    button.button.addEventListener('click', () => {
                        button.play();
                        const timeout = setTimeout(() => {
                            window.postMessage({
                                type: "startPartyNative",
                            }, "*");
                            clearTimeout(timeout);
                        }, 2000);
                        window.addEventListener("message", (event) => {
                            var _a;
                            if (((_a = event.data) === null || _a === void 0 ? void 0 : _a.type) === "startPartyMessageReceived") {
                                clearTimeout(timeout);
                            }
                        });
                        // Clear the click event listener
                        button.button.removeEventListener('click', () => {
                            //
                        });
                    });
                }
            }
        }
        catch (error) {
            // silent catch
        }
    }, 500);
};

;// CONCATENATED MODULE: ./src/Teleparty/BrowseScripts/Amazon/amazon_browse_injected.js

function addNativePartyButton() {
    if (document.getElementById('native-party-button') != null) {
        return undefined;
    }
    const parentDiv = document.querySelector('div[class*="dv-dp-node-playback"]');
    if (parentDiv == null) {
        return undefined;
    }
    const playButton = parentDiv.querySelector('a[class*="fbl-play-btn"]');
    if (playButton == null) {
        return undefined;
    }
    const nativePartyButton = document.createElement('button');
    nativePartyButton.setAttribute('style', 'background: linear-gradient(273.58deg, #9E55A0 0%, #EF3E3A 100%); color: #fff; border-radius: 0.375rem; padding: 1.25rem; margin-top: 8px; display: flex; align-items: center; justify-content: center; text-align: center; cursor: pointer;');
    nativePartyButton.setAttribute('id', 'native-party-button');
    nativePartyButton.innerHTML = '<span>Start a Teleparty</span>';
    playButton.parentElement.parentElement.parentElement.insertBefore(nativePartyButton, playButton.parentElement.parentElement.nextSibling);
    return [{ button: nativePartyButton, play: () => playButton.click() }];
}
addNativePartyHandler(addNativePartyButton);

/******/ })()
;