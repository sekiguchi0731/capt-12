from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.random import Generator

from capt12.mechanisms.baselines import common_cover, k_ary_rr
from capt12.mechanisms.lp import validate_channel


class RandomAdapter(Protocol):
    def choice(self, n: int, p: np.ndarray) -> int: ...


class SecureRNG:
    """Small adapter using rejection-free cumulative sampling from secrets."""

    def choice(self, n: int, p: np.ndarray) -> int:
        value = secrets.randbits(53) / 2**53
        return min(int(np.searchsorted(np.cumsum(p), value, side="right")), n - 1)


@dataclass
class Sanitizer:
    channels: dict[str, np.ndarray]
    token_to_block: np.ndarray
    decoder: np.ndarray
    fallback: str = "common_cover"
    fallback_distribution: np.ndarray | None = None
    fallback_epsilon: float = 0.0
    authenticated: bool = True
    expired: bool = False
    shift_detected: bool = False
    memoize: bool = True
    _memo: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for channel in self.channels.values():
            validate_channel(channel)
        self.token_to_block = np.asarray(self.token_to_block, dtype=int)
        self.decoder = np.asarray(self.decoder, dtype=float)

    def _fallback_channel(self) -> np.ndarray:
        k = len(self.token_to_block)
        if self.fallback == "krr":
            return k_ary_rr(k, self.fallback_epsilon)
        distribution = self.fallback_distribution
        if distribution is None:
            distribution = np.ones(k) / k
        return common_cover(distribution)

    def sanitize(
        self,
        token: int,
        profile: str,
        rng: Generator | RandomAdapter,
        user_epoch_key: str | None = None,
    ) -> int:
        if not 0 <= token < len(self.token_to_block):
            raise ValueError("raw token is outside [0,K)")
        memo_key = None
        if self.memoize and user_epoch_key is not None:
            # Memoization bounds repeated releases within one privacy epoch.
            # The raw token must not be part of the key: otherwise a changing Z
            # would trigger fresh randomized releases and allow composition
            # within the epoch.  The first sanitized output is therefore reused
            # for every later token under the same epoch/profile selection.
            memo_key = hashlib.sha256(f"{user_epoch_key}|{profile}".encode()).hexdigest()
            if memo_key in self._memo:
                return self._memo[memo_key]
        invalid = not self.authenticated or self.expired or self.shift_detected or profile not in self.channels
        if invalid:
            output = int(rng.choice(len(self.token_to_block), p=self._fallback_channel()[token]))
        else:
            source_block = int(self.token_to_block[token])
            destination = int(rng.choice(self.channels[profile].shape[1], p=self.channels[profile][source_block]))
            output = int(rng.choice(len(self.token_to_block), p=self.decoder[destination]))
        if memo_key is not None:
            self._memo[memo_key] = output
        return output


@dataclass
class ContextualSanitizer:
    """Select a certified block channel using only profile and public context."""

    channels: dict[tuple[str, str], np.ndarray]
    token_to_block: np.ndarray
    decoder: np.ndarray
    fallback_distribution: np.ndarray | None = None
    authenticated: bool = True
    expired: bool = False
    memoize: bool = True
    _memo: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.channels = {
            (str(profile), str(context)): np.asarray(channel, dtype=float)
            for (profile, context), channel in self.channels.items()
        }
        for channel in self.channels.values():
            validate_channel(channel)
        self.token_to_block = np.asarray(self.token_to_block, dtype=int)
        self.decoder = np.asarray(self.decoder, dtype=float)

    def _fallback_channel(self) -> np.ndarray:
        distribution = self.fallback_distribution
        if distribution is None:
            distribution = np.ones(len(self.token_to_block)) / len(self.token_to_block)
        return common_cover(distribution)

    def sanitize(
        self,
        token: int,
        profile: str,
        public_context: str,
        rng: Generator | RandomAdapter,
        user_epoch_key: str | None = None,
    ) -> int:
        if not 0 <= token < len(self.token_to_block):
            raise ValueError("raw token is outside [0,K)")
        context = str(public_context)
        memo_key = None
        if self.memoize and user_epoch_key is not None:
            # Public context may select a different certified mechanism, but the
            # protected value and raw token are deliberately absent.  In
            # particular, changes to Z inside an epoch reuse the first release.
            memo_key = hashlib.sha256(
                f"{user_epoch_key}|{profile}|{context}".encode()
            ).hexdigest()
            if memo_key in self._memo:
                return self._memo[memo_key]
        channel = self.channels.get((profile, context))
        invalid = not self.authenticated or self.expired or channel is None
        if invalid:
            output = int(
                rng.choice(
                    len(self.token_to_block),
                    p=self._fallback_channel()[token],
                )
            )
        else:
            source_block = int(self.token_to_block[token])
            destination = int(rng.choice(channel.shape[1], p=channel[source_block]))
            output = int(
                rng.choice(len(self.token_to_block), p=self.decoder[destination])
            )
        if memo_key is not None:
            self._memo[memo_key] = output
        return output
