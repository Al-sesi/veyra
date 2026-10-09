targetScope = 'resourceGroup'

param location string = 'westeurope'
param environmentName string = 'cae-veyra'
param appName string = 'ca-veyra-backend'
param storageAccountName string = 'stveyrahfcache'
param fileShareName string = 'hf-cache'
param logAnalyticsName string = 'la-veyra'

param containerImage string = 'mcr.microsoft.com/azure-app-service/samples/aspnethelloworld:latest'
param containerPort int = 7860

param corsOrigins string = 'https://veyra-1.vercel.app'

@secure()
param databaseUrl string = ''

@secure()
param azureOpenaiApiKey string = ''
param azureOpenaiEndpoint string = ''
param azureOpenaiDeployment string = ''
param azureOpenaiApiVersion string = '2024-08-01-preview'

@secure()
param whatsappAccessToken string = ''
param whatsappPhoneNumberId string = ''
@secure()
param whatsappVerifyToken string = ''
param whatsappBusinessAccountId string = ''

var tags = {
  project: 'veyra'
  component: 'backend'
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  tags: tags
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  name: 'default'
  parent: storageAccount
  properties: {
    shareDeleteRetentionPolicy: {
      enabled: true
      days: 7
    }
  }
}

resource share 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  name: fileShareName
  parent: fileService
  properties: {
    shareQuota: 10
  }
}

var storageAccountKeys = storageAccount.listKeys()

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: containerPort
        transport: 'auto'
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      secrets: [
        {
          name: 'database-url'
          value: databaseUrl
        }
        {
          name: 'azure-openai-api-key'
          value: azureOpenaiApiKey
        }
        {
          name: 'whatsapp-access-token'
          value: whatsappAccessToken
        }
        {
          name: 'whatsapp-verify-token'
          value: whatsappVerifyToken
        }
      ]
      registries: []
    }
    template: {
      volumes: [
        {
          name: 'hf-cache'
          azureFile: {
            accountName: storageAccount.name
            shareName: share.name
            accessKey: storageAccountKeys.keys[0].value
          }
        }
      ]
      containers: [
        {
          image: containerImage
          name: 'veyra-backend'
          resources: {
            cpu: json('2.0')
            memory: '4.0Gi'
          }
          volumeMounts: [
            {
              volumeName: 'hf-cache'
              mountPath: '/data/hf-cache'
            }
          ]
          env: [
            { name: 'PORT', value: '${containerPort}' }
            { name: 'HF_HOME', value: '/data/hf-cache' }
            { name: 'TRANSFORMERS_CACHE', value: '/data/hf-cache' }
            { name: 'VEYRA_CORS_ORIGINS', value: corsOrigins }
            { name: 'DATABASE_URL', secretRef: 'database-url' }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-api-key' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenaiEndpoint }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: azureOpenaiDeployment }
            { name: 'AZURE_OPENAI_API_VERSION', value: azureOpenaiApiVersion }
            { name: 'WHATSAPP_ACCESS_TOKEN', secretRef: 'whatsapp-access-token' }
            { name: 'WHATSAPP_PHONE_NUMBER_ID', value: whatsappPhoneNumberId }
            { name: 'WHATSAPP_VERIFY_TOKEN', secretRef: 'whatsapp-verify-token' }
            { name: 'WHATSAPP_BUSINESS_ACCOUNT_ID', value: whatsappBusinessAccountId }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: containerPort
              }
              periodSeconds: 30
              timeoutSeconds: 10
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: containerPort
              }
              periodSeconds: 15
              timeoutSeconds: 10
            }
            {
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: containerPort
              }
              periodSeconds: 10
              timeoutSeconds: 15
              failureCountThreshold: 30
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
        rules: []
      }
    }
  }
}

output fqdn string = containerApp.properties.configuration.ingress.fqdn
output backendUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output appName string = containerApp.name
output environmentName string = environment.name
output storageAccountName string = storageAccount.name
output fileShareName string = share.name
