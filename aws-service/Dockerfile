FROM node:20-alpine As development

WORKDIR /usr/src/app

COPY package*.json ./

# Using npm ci for reproducible builds
RUN npm ci --omit=dev && npm cache clean --force

###################
# BUILD FOR PRODUCTION
###################

FROM node:20-alpine As build

WORKDIR /usr/src/app

# Copy only necessary production dependencies
COPY --chown=node:node --from=development /usr/src/app/node_modules ./node_modules

# Copy application code
COPY --chown=node:node index.js index.js

USER node

CMD [ "node", "index.js" ]