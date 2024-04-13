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

;// CONCATENATED MODULE: ./src/Teleparty/BrowseScripts/Hulu/hulu_browse_injected.js

function addNativePartyButton() {
    var _a;
    if (document.getElementById('native-party-button') != null) {
        return undefined;
    }
    const playButton = document.querySelector('a[data-testid="watchaction-btn"]');
    if (playButton == null) {
        return undefined;
    }
    const parentDiv = document.querySelector('div[data-testid="watch-action"]').parentElement;
    const nativePartyButton = document.createElement('button');
    nativePartyButton.setAttribute('style', 'background: linear-gradient(273.58deg, #9E55A0 0%, #EF3E3A 100%); color: #fff; border-style: none; border-radius: 0.375rem; padding: 1.25rem; margin-right: 2rem');
    nativePartyButton.setAttribute('id', 'native-party-button');
    const watchMovieText = document.querySelector('div[data-testid="watch-action-text"]');
    const nativePartyButtonText = document.createElement('span');
    nativePartyButtonText.innerText = 'Start a Teleparty';
    nativePartyButtonText.setAttribute('style', ((_a = watchMovieText.getAttribute('style')) !== null && _a !== void 0 ? _a : "") + 'text-align: center; margin-left: 0px; cursor: pointer; white-space: nowrap;');
    nativePartyButtonText.setAttribute('class', watchMovieText.getAttribute('class'));
    parentDiv.insertBefore(nativePartyButton, document.querySelector('div[data-testid="watch-action"]').nextSibling);
    nativePartyButton.appendChild(nativePartyButtonText);
    return [{ button: nativePartyButton, play: () => playButton.click() }];
}
addNativePartyHandler(addNativePartyButton);

/******/ })()
;