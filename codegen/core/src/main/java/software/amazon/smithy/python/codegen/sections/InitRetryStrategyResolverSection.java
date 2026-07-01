/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package software.amazon.smithy.python.codegen.sections;

import software.amazon.smithy.utils.CodeSection;
import software.amazon.smithy.utils.SmithyInternalApi;

/**
 * A section wrapping the construction of the client's {@code RetryStrategyResolver},
 * which integrations may intercept to supply service-specific retry defaults.
 */
@SmithyInternalApi
public record InitRetryStrategyResolverSection() implements CodeSection {}
