import test from 'node:test';
import assert from 'node:assert/strict';
import {clamp} from '../range.js';

test('below range', () => assert.equal(clamp(-3, 0, 10), 0));
test('inside range', () => assert.equal(clamp(4, 0, 10), 4));
test('above range', () => assert.equal(clamp(17, 0, 10), 10));
